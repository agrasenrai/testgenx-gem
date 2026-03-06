# pipeline/stage0/srs_bootstrap.py
#
# Stage 0 — SRS Bootstrap (Gemini-powered)
#
# Reads the GeminiAnalysisResult produced by GeminiSRSAnalyzer and
# populates the SessionKB with ClinicalParam, UserRole, WorkflowDef
# and time-constraint objects.
#
# NO regex patterns.  NO BioGPT.  NO hardcoded clinical values or roles.
# All knowledge comes from Gemini's analysis of THIS specific SRS document.

import logging
from pipeline.session_kb import (
    SessionKB, KnowledgeSource, KnowledgeValue,
    ClinicalParam, UserRole, WorkflowDef,
)
from pipeline.gemini_srs_analyzer import GeminiSRSAnalyzer

logger = logging.getLogger(__name__)


def build_session_kb(
    sentences: list[str],
    skb: SessionKB,
    analyzer: GeminiSRSAnalyzer,
    verbose: bool = False,
) -> None:
    """
    Stage 0: populate SessionKB from Gemini's structured analysis.

    Args:
        sentences : All sentences from the SRS document.
        skb       : SessionKB to populate (mutated in place).
        analyzer  : GeminiSRSAnalyzer with .analyze() already called.
        verbose   : Print detailed extraction results.
    """
    result = analyzer.analyze(sentences)
    kb     = result.knowledge_base

    print("  [Stage0] Mapping Gemini knowledge → SessionKB...")

    # ── 1. Clinical parameters ────────────────────────────────────────────────
    for cp_data in kb.get("clinical_params", []):
        name = cp_data.get("name", "").strip()
        if not name:
            continue

        param = ClinicalParam(
            canonical_name=name,
            unit=cp_data.get("unit", ""),
        )

        def _store_kv(raw_val, evidence_key, cp_data=cp_data, name=name):
            val = cp_data.get(raw_val)
            if val is None:
                return None
            evidence = cp_data.get(evidence_key) or f"Gemini: {raw_val} for '{name}'"
            source   = _infer_source(evidence)
            return KnowledgeValue(
                value=float(val),
                source=source,
                confidence=1.0 if source == KnowledgeSource.SRS_EXTRACTED else 0.85,
                evidence=evidence,
            )

        param.critical_low  = _store_kv("critical_low",  "critical_low_evidence")
        param.critical_high = _store_kv("critical_high", "critical_high_evidence")
        param.normal_low    = _store_kv("normal_low",    "normal_low_evidence")
        param.normal_high   = _store_kv("normal_high",   "normal_high_evidence")

        skb.register_param(param)

        if verbose:
            print(f"    [Param] {name} | unit={param.unit} | "
                  f"crit_low={_kv(param.critical_low)} | "
                  f"crit_high={_kv(param.critical_high)} | "
                  f"norm={_kv(param.normal_low)}–{_kv(param.normal_high)}")

    # ── 2. Roles ──────────────────────────────────────────────────────────────
    for role_data in kb.get("roles", []):
        name = role_data.get("name", "").strip()
        if not name:
            continue

        role = UserRole(
            canonical_name=name,
            authorized=bool(role_data.get("authorized", True)),
            capabilities=role_data.get("capabilities", []),
            access_level=role_data.get("access_level", "unknown"),
            source=KnowledgeSource.SRS_EXTRACTED,
            evidence="Gemini extraction from SRS",
        )
        # Store denied capabilities for Rule Assembler access-control checks
        role.denied_capabilities = role_data.get("denied_capabilities", [])
        skb.register_role(role)

        if verbose:
            auth_str = "AUTHORIZED" if role.authorized else "DENIED"
            print(f"    [Role] {name} | {auth_str} | "
                  f"caps={role.capabilities} | denied={role.denied_capabilities}")

    # ── 3. Workflows ──────────────────────────────────────────────────────────
    for wf_data in kb.get("workflows", []):
        name = wf_data.get("name", "").strip()
        if not name:
            continue

        wf = WorkflowDef(
            name=name,
            states=wf_data.get("states", []),
            valid_transitions=wf_data.get("valid_transitions", []),
            terminal_states=wf_data.get("terminal_states", []),
            trigger_keywords=wf_data.get("trigger_keywords", []),
            source=KnowledgeSource.SRS_EXTRACTED,
        )
        skb.register_workflow(wf)

        if verbose:
            print(f"    [Workflow] {name} | states={wf.states} | "
                  f"transitions={wf.valid_transitions} | "
                  f"terminals={wf.terminal_states}")

    # ── 4. Time constraints ───────────────────────────────────────────────────
    for tc in kb.get("time_constraints", []):
        val  = tc.get("value")
        unit = tc.get("unit", "seconds")
        ctx  = tc.get("context", "")
        if val is not None:
            skb.register_time_constraint(
                value=float(val),
                unit=str(unit).lower(),
                context=ctx,
                source=KnowledgeSource.SRS_EXTRACTED,
            )
            if verbose:
                print(f"    [Time] {val} {unit} | {ctx[:60]}")

    print(f"  [Stage0] Done. {skb.summary()}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_source(evidence: str) -> KnowledgeSource:
    """Mark as SRS_EXTRACTED unless evidence string looks like a model-generated note."""
    if not evidence:
        return KnowledgeSource.CLINICAL_BERT
    lower = evidence.lower()
    if (lower.startswith("gemini:")
            or lower == "clinical knowledge"
            or lower.startswith("clinical ")):
        return KnowledgeSource.CLINICAL_BERT
    return KnowledgeSource.SRS_EXTRACTED


def _kv(kval) -> str:
    if kval is None:
        return "—"
    return f"{kval.value}"
