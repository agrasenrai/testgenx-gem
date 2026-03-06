# pipeline/stage2/scenario_generator.py
# Pure algorithm — no ML models. Generates concrete test scenarios from FormalRules + KB.

import re
from dataclasses import dataclass, field
from pipeline.stage1.rule_assembler import FormalRule, TestStrategy
from pipeline.stage2.knowledge_base import KnowledgeBase
from config import BVA_WELL_INSIDE_OFFSET

# Instrument range maxima for clinical validation (physiologically safe upper bounds)
_INSTRUMENT_MAX = {
    "spo2":                    100,
    "heart_rate":              300,
    "respiratory_rate":         60,
    "glucose":                 600,
    "systolic_blood_pressure": 300,
    "blood_pressure":          300,
}

_scenario_counter = 0


def _next_scenario_id(rule_id: str, suffix: str) -> str:
    global _scenario_counter
    _scenario_counter += 1
    num = rule_id.replace("RULE-", "")
    return f"SC-{num}-{suffix}-{_scenario_counter:03d}"


@dataclass
class Scenario:
    scenario_id: str
    rule_id: str
    strategy: TestStrategy
    scenario_type: str        # e.g. "boundary_below", "valid_class", "invalid_transition"
    inputs: dict = field(default_factory=dict)
    expected_result: str = ""
    priority: str = "MEDIUM"


# ── BVA ─────────────────────────────────────────────────────────────────────

def _apply_bva(rule: FormalRule, kb: KnowledgeBase) -> list:
    scenarios = []
    numeric_conditions = [
        c for c in rule.conditions
        if not c.get("abstract", False) and re.match(r'^\d+\.?\d*$', str(c["value"]))
    ]

    for cond in numeric_conditions:
        try:
            threshold = float(cond["value"])
        except (ValueError, TypeError):
            continue

        param = cond["param"]
        op = cond["operator"]
        base_inputs = {param: None}
        if rule.user_role:
            base_inputs["user_role"] = rule.user_role

        def make_sc(sc_type, value, expected, priority):
            inputs = dict(base_inputs)
            inputs[param] = value
            return Scenario(
                scenario_id=_next_scenario_id(rule.rule_id, f"BVA-{sc_type}"),
                rule_id=rule.rule_id,
                strategy=TestStrategy.BVA,
                scenario_type=sc_type,
                inputs=inputs,
                expected_result=expected,
                priority=priority,
            )

        # For operator "<" (trigger when value < threshold)
        # boundary_exact: value == threshold → condition is FALSE → no trigger
        if op == "<":
            scenarios.append(make_sc("boundary_below", threshold - 1, rule.result, "HIGH"))
            scenarios.append(make_sc("boundary_exact", threshold, f"no_{rule.result}", "HIGH"))
            scenarios.append(make_sc("boundary_above", threshold + 1, f"no_{rule.result}", "MEDIUM"))
            scenarios.append(make_sc("well_inside_valid", max(0, threshold - BVA_WELL_INSIDE_OFFSET), rule.result, "MEDIUM"))
            scenarios.append(make_sc("well_inside_invalid", threshold + BVA_WELL_INSIDE_OFFSET, f"no_{rule.result}", "LOW"))
        # For operator ">" (trigger when value > threshold)
        # boundary_exact: value == threshold → condition is FALSE → no trigger
        elif op == ">":
            scenarios.append(make_sc("boundary_above", threshold + 1, rule.result, "HIGH"))
            scenarios.append(make_sc("boundary_exact", threshold, f"no_{rule.result}", "HIGH"))
            scenarios.append(make_sc("boundary_below", threshold - 1, f"no_{rule.result}", "MEDIUM"))
            scenarios.append(make_sc("well_inside_valid", threshold + BVA_WELL_INSIDE_OFFSET, rule.result, "MEDIUM"))
            scenarios.append(make_sc("well_inside_invalid", max(0, threshold - BVA_WELL_INSIDE_OFFSET), f"no_{rule.result}", "LOW"))
        else:
            # Generic fallback for "="
            scenarios.append(make_sc("exact_match", threshold, rule.result, "HIGH"))
            scenarios.append(make_sc("below_threshold", threshold - 1, f"no_{rule.result}", "MEDIUM"))
            scenarios.append(make_sc("above_threshold", threshold + 1, f"no_{rule.result}", "MEDIUM"))

    return scenarios


# ── EP ──────────────────────────────────────────────────────────────────────

def _apply_ep(rule: FormalRule, kb: KnowledgeBase) -> list:
    scenarios = []

    authorized_roles   = kb.get_authorized_roles()
    unauthorized_roles = kb.get_unauthorized_roles()

    # Valid class: always use the role the SRS explicitly names for this rule.
    # If no role on the rule, take the first authorized role from the KB.
    # Only skip if KB is truly empty (no roles extracted from this SRS at all).
    valid_role = rule.user_role if rule.user_role else (
        authorized_roles[0] if authorized_roles else None
    )
    # Invalid class: first explicitly unauthorized role from KB.
    invalid_role = unauthorized_roles[0] if unauthorized_roles else None

    if not valid_role and not invalid_role:
        return scenarios   # no role data in KB — cannot generate meaningful EP TCs

    base_inputs = {}
    # Include any BVA-like numeric values from conditions for context
    for cond in rule.conditions:
        if not cond.get("abstract", False) and re.match(r'^\d+\.?\d*$', str(cond["value"])):
            base_inputs[cond["param"]] = float(cond["value"])

    def make_ep(sc_type, role, expected, priority):
        inputs = dict(base_inputs)
        inputs["user_role"] = role
        return Scenario(
            scenario_id=_next_scenario_id(rule.rule_id, f"EP-{sc_type}"),
            rule_id=rule.rule_id,
            strategy=TestStrategy.EP,
            scenario_type=sc_type,
            inputs=inputs,
            expected_result=expected,
            priority=priority,
        )

    if valid_role:
        scenarios.append(make_ep("valid_class",   valid_role,   rule.result,    "MEDIUM"))
    if invalid_role:
        scenarios.append(make_ep("invalid_class", invalid_role, "access_denied", "HIGH"))

    # Boundary class: KB roles with intermediate access ("limited" or "read-only"),
    # excluding the roles already used above.
    used = {valid_role, invalid_role}
    all_kb_roles = kb.get_all_roles() if hasattr(kb, "get_all_roles") else []
    for r in all_kb_roles:
        if r not in used and r not in (valid_role, invalid_role):
            # Use it as a boundary class test if KB has it.
            scenarios.append(make_ep("boundary_class", r, "access_controlled", "MEDIUM"))
            break  # one boundary case is sufficient

    return scenarios


# ── State Transition ─────────────────────────────────────────────────────────


def _apply_state_transition(rule: FormalRule, kb: KnowledgeBase) -> list:
    scenarios = []

    # Use SessionKB's own KB-driven workflow detector (scores by trigger_keywords
    # and state names already registered from the SRS — no hardcoded word lists).
    wf = None
    workflow_name = None

    detected_name = (
        kb.detect_workflow_from_text(rule.source_sentence)
        if hasattr(kb, "detect_workflow_from_text")
        else None
    )
    if detected_name:
        wf = kb.get_workflow(detected_name)
        workflow_name = detected_name

    # Fallback: first available workflow in KB
    if not wf:
        for kn in kb.get_all_workflow_names():
            wf = kb.get_workflow(kn)
            if wf:
                workflow_name = kn
                break

    if not wf:
        return scenarios

    # Support both dict (old KnowledgeBase) and WorkflowDef dataclass (SessionKB)
    if isinstance(wf, dict):
        states = wf.get("states", [])
        valid_transitions = [tuple(t) for t in wf.get("valid_transitions", [])]
        terminal_states = wf.get("terminal_states", [])
    else:
        states = wf.states
        valid_transitions = [tuple(t) for t in wf.valid_transitions]
        terminal_states = wf.terminal_states

    # Valid transitions → should succeed
    for (from_state, to_state) in valid_transitions:
        sc = Scenario(
            scenario_id=_next_scenario_id(rule.rule_id, f"ST-valid"),
            rule_id=rule.rule_id,
            strategy=TestStrategy.STATE_TRANSITION,
            scenario_type="valid_transition",
            inputs={"workflow": workflow_name, "from_state": from_state, "to_state": to_state,
                    "user_role": rule.user_role},
            expected_result="transition_success",
            priority="MEDIUM",
        )
        scenarios.append(sc)

    # Invalid transitions (skip states)
    if len(states) >= 3:
        skip_transitions = []
        for i, s_from in enumerate(states[:-2]):
            for s_to in states[i + 2:]:
                if (s_from, s_to) not in valid_transitions:
                    skip_transitions.append((s_from, s_to))
        for (from_state, to_state) in skip_transitions[:3]:  # limit to 3
            sc = Scenario(
                scenario_id=_next_scenario_id(rule.rule_id, f"ST-invalid-skip"),
                rule_id=rule.rule_id,
                strategy=TestStrategy.STATE_TRANSITION,
                scenario_type="invalid_transition_skip",
                inputs={"workflow": workflow_name, "from_state": from_state, "to_state": to_state,
                        "user_role": rule.user_role},
                expected_result="transition_rejected",
                priority="HIGH",
            )
            scenarios.append(sc)

    # Transitions from terminal states → should fail
    non_terminal = [s for s in states if s not in terminal_states]
    for term_state in terminal_states:
        target = non_terminal[0] if non_terminal else states[0]
        sc = Scenario(
            scenario_id=_next_scenario_id(rule.rule_id, f"ST-from-terminal"),
            rule_id=rule.rule_id,
            strategy=TestStrategy.STATE_TRANSITION,
            scenario_type="invalid_transition_from_terminal",
            inputs={"workflow": workflow_name, "from_state": term_state, "to_state": target,
                    "user_role": rule.user_role},
            expected_result="transition_rejected",
            priority="HIGH",
        )
        scenarios.append(sc)

    return scenarios


# ── Decision Table ────────────────────────────────────────────────────────────────────────────

def _apply_decision_table(rule: FormalRule, kb: KnowledgeBase) -> list:
    scenarios = []

    # Build list of binary conditions from rule
    binary_conditions = []
    for cond in rule.conditions:
        if not cond.get("abstract", False):
            binary_conditions.append({
                "name": cond["param"],
                "true_value": float(cond["value"]) - 1 if cond["operator"] == "<" else float(cond["value"]) + 1,
                "false_value": float(cond["value"]) + 1 if cond["operator"] == "<" else float(cond["value"]) - 1,
            })

    if rule.user_role:
        authorized   = kb.get_authorized_roles()
        unauthorized = kb.get_unauthorized_roles()
        # Use rule.user_role as the "true" value; use first unauthorized as "false".
        # Only add role condition if unauthorized roles exist in KB.
        if unauthorized:
            binary_conditions.append({
                "name":        "user_role",
                "true_value":  rule.user_role,
                "false_value": unauthorized[0],
            })

    if not binary_conditions:
        return scenarios

    n = len(binary_conditions)

    def make_dt(sc_type, values_dict, expected, priority):
        return Scenario(
            scenario_id=_next_scenario_id(rule.rule_id, f"DT-{sc_type}"),
            rule_id=rule.rule_id,
            strategy=TestStrategy.DECISION_TABLE,
            scenario_type=sc_type,
            inputs=values_dict,
            expected_result=expected,
            priority=priority,
        )

    # All-true → should trigger
    all_true = {c["name"]: c["true_value"] for c in binary_conditions}
    scenarios.append(make_dt("all_conditions_true", all_true, rule.result, "HIGH"))

    # All-false → should NOT trigger
    all_false = {c["name"]: c["false_value"] for c in binary_conditions}
    scenarios.append(make_dt("all_conditions_false", all_false, f"no_{rule.result}", "MEDIUM"))

    # For 2 conditions: all 4 combos
    if n == 2:
        tf = {binary_conditions[0]["name"]: binary_conditions[0]["true_value"],
              binary_conditions[1]["name"]: binary_conditions[1]["false_value"]}
        ft = {binary_conditions[0]["name"]: binary_conditions[0]["false_value"],
              binary_conditions[1]["name"]: binary_conditions[1]["true_value"]}
        scenarios.append(make_dt("cond1_true_cond2_false", tf, f"partial_{rule.result}", "MEDIUM"))
        scenarios.append(make_dt("cond1_false_cond2_true", ft, f"partial_{rule.result}", "MEDIUM"))
    else:
        # For 3+ conditions: single-flip scenarios
        for i, cond in enumerate(binary_conditions[:n]):
            flipped = dict(all_true)
            flipped[cond["name"]] = cond["false_value"]
            scenarios.append(make_dt(f"flip_cond_{i+1}", flipped, f"no_{rule.result}", "MEDIUM"))

    return scenarios


# ── Temporal ─────────────────────────────────────────────────────────────────

def _apply_temporal(rule: FormalRule, kb) -> list:
    """
    Generate timing/deadline test scenarios for requirements like
    'within 5 seconds', 'within 10 minutes', 'within 24 hours'.

    Produces 4 scenarios: well within → just before → at deadline → exceeded.
    """
    scenarios = []
    tc_raw = str(rule.time_constraint or "").strip()
    if not tc_raw:
        return scenarios

    m = re.search(r'(\d+\.?\d*)\s*(second|minute|hour|day)', tc_raw, re.IGNORECASE)
    if not m:
        return scenarios

    deadline = float(m.group(1))
    unit     = m.group(2).lower() + ("s" if not m.group(2).lower().endswith("s") else "")

    half       = round(deadline * 0.5, 1)
    just_under = round(deadline - (1 if "second" in unit else 0.5), 1)
    just_over  = round(deadline + (1 if "second" in unit else 1), 1)

    base_inputs = {"unit": unit}
    if rule.user_role:
        base_inputs["user_role"] = rule.user_role

    def make_tmp(sc_type, time_val, expected, priority):
        inp = dict(base_inputs)
        inp["response_time_measured"] = time_val
        return Scenario(
            scenario_id=_next_scenario_id(rule.rule_id, f"TMP-{sc_type}"),
            rule_id=rule.rule_id,
            strategy=TestStrategy.TEMPORAL,
            scenario_type=sc_type,
            inputs=inp,
            expected_result=expected,
            priority=priority,
        )

    scenarios.append(make_tmp("well_within_deadline",  half,             f"{rule.result}_compliant", "LOW"))
    scenarios.append(make_tmp("just_before_deadline",  max(0.1, just_under), f"{rule.result}_compliant", "MEDIUM"))
    scenarios.append(make_tmp("at_deadline_boundary",  deadline,         f"boundary_{rule.result}",  "HIGH"))
    scenarios.append(make_tmp("deadline_exceeded",     just_over,        "non_compliant_timeout",    "HIGH"))
    return scenarios


# ── Clinical Validation ───────────────────────────────────────────────────────

def _apply_clinical_validation(rule: FormalRule, kb) -> list:
    """
    Generate invalid-input test scenarios: null, non-numeric, negative,
    above instrument maximum, and a valid-normal baseline.
    Targets clinical realism and safety validation.
    """
    scenarios = []
    numeric_conditions = [
        c for c in rule.conditions
        if not c.get("abstract", False)
        and re.match(r'^\d+\.?\d*$', str(c.get("value", "")))
    ]
    if not numeric_conditions:
        return scenarios

    cond      = numeric_conditions[0]
    param     = cond["param"]
    threshold = float(cond["value"])
    op        = cond.get("operator", "<")

    inst_max  = _INSTRUMENT_MAX.get(param, int(threshold * 10))
    if op == "<":
        normal_val = round(threshold + BVA_WELL_INSIDE_OFFSET, 1)
    elif op == ">":
        normal_val = round(threshold - BVA_WELL_INSIDE_OFFSET, 1)
    else:
        normal_val = round(threshold + 1, 1)

    base_inputs = {}
    if rule.user_role:
        base_inputs["user_role"] = rule.user_role

    def make_cv(sc_type, value, expected, priority):
        inp = dict(base_inputs)
        inp[param] = value
        return Scenario(
            scenario_id=_next_scenario_id(rule.rule_id, f"CV-{sc_type}"),
            rule_id=rule.rule_id,
            strategy=TestStrategy.CLINICAL_VALIDATION,
            scenario_type=sc_type,
            inputs=inp,
            expected_result=expected,
            priority=priority,
        )

    scenarios.append(make_cv("null_input",            None,          "validation_error_null_rejected",        "HIGH"))
    scenarios.append(make_cv("non_numeric_type",      "abc",         "validation_error_type_mismatch",        "HIGH"))
    scenarios.append(make_cv("negative_value",        -1,            "validation_error_below_instrument_min", "HIGH"))
    scenarios.append(make_cv("above_instrument_max",  inst_max + 50, "validation_error_above_instrument_max", "HIGH"))
    scenarios.append(make_cv("valid_normal_baseline", normal_val,    "accepted_no_alert",                     "MEDIUM"))
    return scenarios


# ── Main Entry Point ─────────────────────────────────────────────────────────

def generate_scenarios(rules: list, kb) -> list:
    """
    Apply all applicable test strategies to each FormalRule and return all Scenarios.

    Args:
        rules: List of FormalRule objects.
        kb: Populated KnowledgeBase or SessionKB.

    Returns:
        List of Scenario objects.
    """
    global _scenario_counter
    _scenario_counter = 0

    all_scenarios = []

    for rule in rules:
        rule_scenarios = []

        for strategy in rule.strategies:
            try:
                if strategy == TestStrategy.BVA:
                    rule_scenarios.extend(_apply_bva(rule, kb))
                elif strategy == TestStrategy.EP:
                    rule_scenarios.extend(_apply_ep(rule, kb))
                elif strategy == TestStrategy.STATE_TRANSITION:
                    rule_scenarios.extend(_apply_state_transition(rule, kb))
                elif strategy == TestStrategy.DECISION_TABLE:
                    rule_scenarios.extend(_apply_decision_table(rule, kb))
                elif strategy == TestStrategy.TEMPORAL:
                    rule_scenarios.extend(_apply_temporal(rule, kb))
                elif strategy == TestStrategy.CLINICAL_VALIDATION:
                    rule_scenarios.extend(_apply_clinical_validation(rule, kb))
            except Exception as e:
                print(f"  [WARN] Strategy {strategy} failed for {rule.rule_id}: {e}")

        # Guarantee at least one scenario per rule.
        # Only produce a fallback EP scenario if we have a role from KB or the rule.
        if not rule_scenarios:
            authorized    = kb.get_authorized_roles()
            fallback_role = rule.user_role or (authorized[0] if authorized else None)
            if fallback_role:
                rule_scenarios.append(Scenario(
                    scenario_id=_next_scenario_id(rule.rule_id, "EP-default"),
                    rule_id=rule.rule_id,
                    strategy=TestStrategy.EP,
                    scenario_type="default",
                    inputs={"user_role": fallback_role},
                    expected_result=rule.result,
                    priority="MEDIUM",
                ))

        all_scenarios.extend(rule_scenarios)

    return all_scenarios
