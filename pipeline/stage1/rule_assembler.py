# pipeline/stage1/rule_assembler.py
#
# Converts NERResult objects into FormalRule objects.
# KEY CHANGE: Any condition referencing a clinical param that is NOT
# in the SessionKB (unresolvable) causes that condition to be DROPPED.
# If a rule ends up with zero resolvable conditions AND no role/strategy,
# the entire rule is DROPPED and logged in the SessionKB skip log.
# Zero unresolved rules reach Stage 2.

import re
from dataclasses import dataclass, field
from enum import Enum

from pipeline.stage1.ner_extractor import NERResult
from pipeline.session_kb import SessionKB


class TestStrategy(Enum):
    BVA                  = "BVA"
    EP                   = "EP"
    STATE_TRANSITION     = "STATE_TRANSITION"
    DECISION_TABLE       = "DECISION_TABLE"
    TEMPORAL             = "TEMPORAL"             # tests timing/deadline requirements
    CLINICAL_VALIDATION  = "CLINICAL_VALIDATION"  # tests invalid/null/out-of-range inputs


@dataclass
class FormalRule:
    rule_id: str
    source_sentence: str
    action: str
    conditions: list = field(default_factory=list)
    # Each: {param, operator, value, unit, abstract, source}
    user_role: object = None
    result: str = "verified"
    time_constraint: object = None
    strategies: list = field(default_factory=list)
    has_abstract_terms: bool = False


_ACTION_TO_RESULT = {
    "alert":      "alert_triggered",
    "notify":     "alert_triggered",
    "restrict":   "access_denied",
    "prevent":    "action_rejected",
    "log":        "audit_log_entry_created",
    "record":     "audit_log_entry_created",
    "display":    "warning_displayed",
    "require":    "authentication_required",
    "generate":   "alert_generated",
    "escalate":   "alert_escalated",
    "lock":       "session_locked",
    "transition": "state_transitioned",
}


def _infer_result(action: str) -> str:
    return _ACTION_TO_RESULT.get(action, "action_completed")


def _parse_value_and_unit(value_str: str) -> tuple:
    m = re.match(r'^(\d+\.?\d*)\s*(.*)', value_str.strip())
    if m:
        try:
            return float(m.group(1)), m.group(2).strip()
        except ValueError:
            pass
    return None, ""


_rule_counter = 0


def _next_rule_id() -> str:
    global _rule_counter
    _rule_counter += 1
    return f"RULE-{_rule_counter:03d}"


def _detect_strategies(sentence: str, conditions: list,
                        user_role: str | None, condition_keywords: list,
                        skb: SessionKB,
                        time_constraint=None) -> list[TestStrategy]:
    strategies = []
    sent_lower = sentence.lower()

    # BVA: has at least one numeric resolvable condition
    numeric = [
        c for c in conditions
        if not c.get("abstract") and c.get("value") is not None
        and re.match(r'^\d+\.?\d*$', str(c["value"]))
    ]
    if numeric:
        strategies.append(TestStrategy.BVA)

    # EP: role context
    role_aliases = skb.get_all_role_aliases_flat()
    has_role_context = (
        user_role is not None
        or any(alias in sent_lower for alias in role_aliases)
        or "access" in sent_lower
        or "permission" in sent_lower
        or "unauthorized" in sent_lower
        or "authorized" in sent_lower
    )
    if has_role_context:
        strategies.append(TestStrategy.EP)

    # STATE_TRANSITION: state-change indicator words from SessionKB
    if any(w in sent_lower for w in skb.state_transition_indicator_words):
        strategies.append(TestStrategy.STATE_TRANSITION)

    # DECISION_TABLE: multiple conditions or multiple condition keywords
    if len(conditions) > 1 or len(condition_keywords) > 1:
        strategies.append(TestStrategy.DECISION_TABLE)

    # TEMPORAL: sentence has an explicit time deadline (within N / after N)
    if time_constraint:
        strategies.append(TestStrategy.TEMPORAL)

    # CLINICAL_VALIDATION: has at least one SRS-sourced numeric condition
    # — generates null/type/range/boundary invalid-input TCs
    srs_numeric = [
        c for c in numeric
        if c.get("source") in ("srs_extracted", "event_based", None)
    ]
    if srs_numeric:
        strategies.append(TestStrategy.CLINICAL_VALIDATION)

    if not strategies:
        strategies.append(TestStrategy.EP)

    return strategies


def assemble_rules(ner_results: list, skb: SessionKB) -> list:
    """
    Convert NERResult objects into FormalRule objects.
    Rules with unresolvable params are dropped and logged.

    Args:
        ner_results: From ner_extractor.extract_entities().
        skb:         SessionKB — used for param/role resolution and skip logging.

    Returns:
        List of FormalRule objects (all fully resolvable).
    """
    global _rule_counter
    _rule_counter = 0
    rules = []

    for ner in ner_results:
        sentence = ner.sentence
        entities = ner.entities

        # ── Primary action ────────────────────────────────────────────────────
        actions = entities.get("ACTION", [])
        primary_action = actions[0] if actions else "verify"

        # ── User role ─────────────────────────────────────────────────────────
        roles_found = entities.get("USER_ROLE", [])
        user_role = None
        has_abstract = False

        for rc in roles_found:
            if skb.is_abstract_role(rc):
                has_abstract = True
                expanded = skb.expand_abstract_role(rc)
                if expanded:
                    user_role = expanded[0]
            else:
                resolved = skb.resolve_role_name(rc)
                user_role = resolved if resolved else rc
                break

        # ── Time constraint ───────────────────────────────────────────────────
        time_constraints = entities.get("TIME_CONSTRAINT", [])
        time_constraint = time_constraints[0] if time_constraints else None

        # ── Build conditions ───────────────────────────────────────────────────
        params    = entities.get("CLINICAL_PARAM", [])
        operators = entities.get("OPERATOR", [])
        values    = entities.get("CLINICAL_VALUE", [])

        conditions = []
        for i in range(max(len(params), len(values)) if (params or values) else 0):
            raw_param = params[i] if i < len(params) else (params[0] if params else "parameter")
            raw_op    = operators[i] if i < len(operators) else (operators[0] if operators else "=")
            raw_val   = values[i] if i < len(values) else "detected"

            # Resolve param via SessionKB — event-based params (drug_interaction etc.)
            # may not be in the KB; treat them as event-based but still usable.
            canonical_param = skb.resolve_param_name(str(raw_param))
            kb_param        = skb.get_param(canonical_param) if canonical_param else None

            if canonical_param is None:
                # Not in KB — use as-is (event/boolean param, EP strategy applies)
                canonical_param = str(raw_param).strip().lower().replace(" ", "_")
                kb_param        = None

            # Operator
            raw_op_sym = raw_op if raw_op in ("<", ">", "=", "!=", ">=", "<=") else (
                skb.resolve_operator(raw_op) or "="
            )

            # Value
            numeric_val, unit = _parse_value_and_unit(str(raw_val))
            condition_value = str(numeric_val) if numeric_val is not None else str(raw_val)

            abstract_val = skb.is_abstract_role(condition_value)
            if abstract_val:
                has_abstract = True

            source_tag = "srs_extracted"
            if kb_param and kb_param.critical_low and kb_param.critical_low.source.value == "clinical_bert":
                source_tag = "clinical_bert"
            elif kb_param is None:
                source_tag = "event_based"

            conditions.append({
                "param":    canonical_param,
                "operator": raw_op_sym,
                "value":    condition_value,
                "unit":     unit,
                "abstract": abstract_val,
                "source":   source_tag,
            })

        # ── Drop rule if it has no usable content ─────────────────────────────
        has_state_transition = any(
            w in sentence.lower() for w in skb.state_transition_indicator_words
        )
        has_role_context  = user_role is not None or "access" in sentence.lower()
        has_action_trigger = bool(primary_action != "verify" and entities.get("CONDITION"))

        if not conditions and not has_role_context and not has_state_transition and not has_action_trigger:
            rule_id = _next_rule_id()
            skb.log_skipped_rule(
                rule_id=rule_id,
                reason="No resolvable conditions, no role context, no state-transition keywords",
                sentence=sentence,
            )
            continue

        # ── Synthesize boolean condition for action-only rules ─────────────────
        # When Gemini has ACTION + CONDITION but no CLINICAL_PARAM, create a
        # boolean event condition so scenario_generator has something to work with.
        if not conditions and has_action_trigger:
            event_param = f"{primary_action}_trigger"
            conditions.append({
                "param":    event_param,
                "operator": "=",
                "value":    "detected",
                "unit":     "",
                "abstract": False,
                "source":   "event_based",
            })

        # ── Infer result (prefer Gemini's specific description) ──────────────
        result = (
            ner.expected_result
            if getattr(ner, "expected_result", "").strip()
            else _infer_result(primary_action)
        )

        # ── Strategies ───────────────────────────────────────────────────────
        condition_keywords = entities.get("CONDITION", [])
        strategies = _detect_strategies(sentence, conditions, user_role, condition_keywords, skb,
                                        time_constraint=time_constraint)

        rules.append(FormalRule(
            rule_id=_next_rule_id(),
            source_sentence=sentence,
            action=primary_action,
            conditions=conditions,
            user_role=user_role,
            result=result,
            time_constraint=time_constraint,
            strategies=strategies,
            has_abstract_terms=has_abstract,
        ))

    return rules


import logging
logger = logging.getLogger(__name__)