# pipeline/session_kb.py
#
# The single context object built for one SRS document run.
# Every piece of information has a source tag.
# Only two valid sources: SRS_EXTRACTED or CLINICAL_BERT.
# Nothing hardcoded. Nothing anonymous. Nothing unresolved reaches Stage 1+.

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class KnowledgeSource(Enum):
    SRS_EXTRACTED = "srs_extracted"   # found directly in the SRS text
    CLINICAL_BERT = "clinical_bert"   # answered by BioGPT / PubMedBERT


@dataclass
class KnowledgeValue:
    value: Any
    source: KnowledgeSource
    confidence: float
    evidence: str = ""   # the SRS sentence it came from, or the BioGPT query


@dataclass
class ClinicalParam:
    """
    A clinical measurable parameter extracted from the SRS.
    Threshold values come from either the SRS text or the clinical model.
    """
    canonical_name: str
    unit: str = ""
    param_type: str = "float"           # "integer" or "float"
    aliases: list = field(default_factory=list)
    critical_low:  KnowledgeValue | None = None
    critical_high: KnowledgeValue | None = None
    normal_low:    KnowledgeValue | None = None
    normal_high:   KnowledgeValue | None = None

    def is_fully_resolved(self) -> bool:
        """True if we have at least critical_low OR critical_high for BVA."""
        return self.critical_low is not None or self.critical_high is not None

    def get_bva_threshold(self) -> tuple[float | None, str]:
        """Return (threshold_value, operator) for BVA, or (None, '') if unresolved."""
        if self.critical_low is not None:
            return self.critical_low.value, "<"
        if self.critical_high is not None:
            return self.critical_high.value, ">"
        return None, ""


@dataclass
class UserRole:
    """
    A user role extracted from the SRS.
    Authorization is always sourced from SRS text — never guessed.
    """
    canonical_name: str
    authorized: bool
    capabilities: list = field(default_factory=list)
    access_level: str = "unknown"
    aliases: list = field(default_factory=list)
    source: KnowledgeSource = KnowledgeSource.SRS_EXTRACTED
    evidence: str = ""


@dataclass
class WorkflowDef:
    """
    A workflow state machine extracted from the SRS.
    States and transitions are sourced from SRS text.
    """
    name: str
    states: list = field(default_factory=list)
    valid_transitions: list = field(default_factory=list)   # list of [from, to] pairs
    terminal_states: list = field(default_factory=list)
    trigger_keywords: list = field(default_factory=list)
    source: KnowledgeSource = KnowledgeSource.SRS_EXTRACTED


class SessionKB:
    """
    Self-building knowledge base for one SRS document run.

    Built by Stage 0 (SRS bootstrap + Clinical BERT gap filler).
    Read and extended by Stages 1, 2, 3.

    Rules:
     - Every value has a KnowledgeSource tag.
     - Params without a resolved threshold are SKIPPED (never stored here).
     - No static fallback. No anonymous defaults. No unresolved values.
    """

    def __init__(self, linguistic_registry_path: str):
        reg_path = Path(linguistic_registry_path)
        if not reg_path.exists():
            raise FileNotFoundError(f"Linguistic registry not found: {linguistic_registry_path}")
        with open(reg_path, "r", encoding="utf-8") as f:
            self._registry = json.load(f)

        # Build fast NLP lookup indices from the registry
        self._build_linguistic_indices()

        # ── Clinical knowledge (built from SRS + BioGPT) ─────────────────────
        self._params:    dict[str, ClinicalParam] = {}
        self._roles:     dict[str, UserRole]      = {}
        self._workflows: dict[str, WorkflowDef]   = {}
        self._time_constraints: list[dict]         = []

        # ── Skip log (transparency) ───────────────────────────────────────────
        self.skipped_params: list[dict] = []   # params that were unresolvable
        self.skipped_rules:  list[dict] = []   # rules dropped due to unresolvable params

        # ── Pipeline state (grows as pipeline runs) ───────────────────────────
        self.all_sentences:          list = []
        self.classified_sentences:   list = []
        self.ner_results:            list = []
        self.formal_rules:           list = []
        self.scenarios:              list = []
        self.test_cases:             list = []

    # ── Linguistic index builders ─────────────────────────────────────────────

    def _build_linguistic_indices(self):
        reg = self._registry

        # operator surface form (lower) → symbol
        self._operator_index: dict[str, str] = {}
        for sym, forms in reg.get("operator_surface_forms", {}).items():
            for f in forms:
                self._operator_index[f.lower()] = sym

        # action surface form (lower) → canonical action
        self._action_index: dict[str, str] = {}
        for canon, forms in reg.get("action_surface_forms", {}).items():
            for f in forms:
                self._action_index[f.lower()] = canon

        self._condition_words: list[str] = reg.get("condition_trigger_words", [])
        self._state_transition_words: list[str] = reg.get("state_transition_indicator_words", [])
        self._abstract_role_markers: list[str] = reg.get("abstract_role_markers", [])
        self._ner_group_map: dict = reg.get("biomedical_ner_group_to_canonical", {})
        self._dep_action_roles:  list[str] = reg.get("spacy_dep_action_roles", [])
        self._dep_subject_roles: list[str] = reg.get("spacy_dep_subject_roles", [])
        self._dep_object_roles:  list[str] = reg.get("spacy_dep_object_roles", [])

    # ── Registration (called by Stage 0) ─────────────────────────────────────

    def register_param(self, param: ClinicalParam) -> None:
        """Only stores params that are fully resolved. Logs skips otherwise."""
        if param.is_fully_resolved():
            key = param.canonical_name.lower()
            # Merge with existing if already registered
            if key in self._params:
                existing = self._params[key]
                if param.critical_low  and not existing.critical_low:
                    existing.critical_low  = param.critical_low
                if param.critical_high and not existing.critical_high:
                    existing.critical_high = param.critical_high
                if param.normal_low    and not existing.normal_low:
                    existing.normal_low    = param.normal_low
                if param.normal_high   and not existing.normal_high:
                    existing.normal_high   = param.normal_high
            else:
                self._params[key] = param
        else:
            self.skipped_params.append({
                "param": param.canonical_name,
                "reason": "No threshold found in SRS or clinical model",
            })

    def register_role(self, role: UserRole) -> None:
        self._roles[role.canonical_name.lower()] = role

    def register_workflow(self, wf: WorkflowDef) -> None:
        self._workflows[wf.name.lower()] = wf

    def register_time_constraint(self, value: float, unit: str,
                                  context: str = "", source: KnowledgeSource = KnowledgeSource.SRS_EXTRACTED):
        self._time_constraints.append({
            "value": value, "unit": unit,
            "context": context, "source": source.value,
        })

    def log_skipped_rule(self, rule_id: str, reason: str, sentence: str):
        self.skipped_rules.append({
            "rule_id": rule_id, "reason": reason, "sentence": sentence
        })

    # ── Param lookups ─────────────────────────────────────────────────────────

    def get_param(self, name: str) -> ClinicalParam | None:
        return self._params.get(name.lower())

    def get_all_param_names(self) -> list[str]:
        return [p.canonical_name for p in self._params.values()]

    def resolve_param_name(self, text: str) -> str | None:
        """Match text against canonical names and aliases. Returns canonical name."""
        tl = text.lower().strip()
        # Direct match on canonical
        if tl in self._params:
            return self._params[tl].canonical_name
        # Match on aliases
        best, best_len = None, 0
        for key, param in self._params.items():
            if tl == key and len(key) > best_len:
                best, best_len = param.canonical_name, len(key)
            for alias in param.aliases:
                if alias.lower() in tl and len(alias) > best_len:
                    best, best_len = param.canonical_name, len(alias)
        return best

    def get_all_param_aliases_flat(self) -> list[str]:
        aliases = []
        for param in self._params.values():
            aliases.append(param.canonical_name.lower())
            aliases.extend(a.lower() for a in param.aliases)
        return sorted(set(aliases), key=len, reverse=True)

    # ── Role lookups ──────────────────────────────────────────────────────────

    def get_all_roles(self) -> list[str]:
        return [r.canonical_name for r in self._roles.values()]

    def get_authorized_roles(self) -> list[str]:
        return [r.canonical_name for r in self._roles.values() if r.authorized]

    def get_unauthorized_roles(self) -> list[str]:
        return [r.canonical_name for r in self._roles.values() if not r.authorized]

    def get_boundary_roles(self) -> list[str]:
        """Authorized roles with limited (non-full) access — EP boundary class."""
        return [
            r.canonical_name for r in self._roles.values()
            if r.authorized and r.access_level not in ("full", "admin", "unknown")
        ]

    def get_role(self, name: str) -> UserRole | None:
        return self._roles.get(name.lower())

    def resolve_role_name(self, text: str) -> str | None:
        tl = text.lower().strip()
        if tl in self._roles:
            return self._roles[tl].canonical_name
        best, best_len = None, 0
        for key, role in self._roles.items():
            if key in tl and len(key) > best_len:
                best, best_len = role.canonical_name, len(key)
            for alias in role.aliases:
                if alias.lower() in tl and len(alias) > best_len:
                    best, best_len = role.canonical_name, len(alias)
        return best

    def get_all_role_aliases_flat(self) -> list[str]:
        aliases = []
        for role in self._roles.values():
            aliases.append(role.canonical_name.lower())
            aliases.extend(a.lower() for a in role.aliases)
        return sorted(set(aliases), key=len, reverse=True)

    def is_abstract_role(self, text: str) -> bool:
        tl = text.lower()
        return any(m in tl for m in self._abstract_role_markers)

    def expand_abstract_role(self, text: str) -> list[str]:
        tl = text.lower()
        if "unauthorized" in tl or "not authorized" in tl:
            return self.get_unauthorized_roles()
        if "authorized" in tl or "valid" in tl:
            return self.get_authorized_roles()
        return self.get_all_roles()

    # ── Workflow lookups ──────────────────────────────────────────────────────

    def get_workflow(self, name: str) -> WorkflowDef | None:
        return self._workflows.get(name.lower())

    def get_all_workflow_names(self) -> list[str]:
        return [w.name for w in self._workflows.values()]

    def detect_workflow_from_text(self, text: str) -> str | None:
        tl = text.lower()
        scores: dict[str, int] = {}
        for key, wf in self._workflows.items():
            score = sum(1 for kw in wf.trigger_keywords if kw in tl)
            # Also score against state names
            score += sum(1 for s in wf.states if s.lower() in tl)
            if score > 0:
                scores[key] = score
        return max(scores, key=scores.__getitem__) if scores else None

    # ── Linguistic lookup (used by NER) ───────────────────────────────────────

    def resolve_operator(self, text: str) -> str | None:
        tl = text.lower().strip()
        if tl in self._operator_index:
            return self._operator_index[tl]
        best, best_len = None, 0
        for form, sym in self._operator_index.items():
            if form in tl and len(form) > best_len:
                best, best_len = sym, len(form)
        return best

    def resolve_action(self, text: str) -> str | None:
        tl = text.lower().strip()
        if tl in self._action_index:
            return self._action_index[tl]
        best, best_len = None, 0
        for form, canon in self._action_index.items():
            if form in tl and len(form) > best_len:
                best, best_len = canon, len(form)
        return best

    def map_ner_group(self, group: str) -> str | None:
        return self._ner_group_map.get(group)

    @property
    def condition_trigger_words(self) -> list[str]:
        return self._condition_words

    @property
    def state_transition_indicator_words(self) -> list[str]:
        return self._state_transition_words

    @property
    def spacy_dep_action_roles(self) -> list[str]:
        return self._dep_action_roles

    @property
    def spacy_dep_subject_roles(self) -> list[str]:
        return self._dep_subject_roles

    @property
    def spacy_dep_object_roles(self) -> list[str]:
        return self._dep_object_roles

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "params_resolved":    len(self._params),
            "params_skipped":     len(self.skipped_params),
            "roles_extracted":    len(self._roles),
            "workflows_extracted": len(self._workflows),
            "time_constraints":   len(self._time_constraints),
            "rules_skipped":      len(self.skipped_rules),
        }