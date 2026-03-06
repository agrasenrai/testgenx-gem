# pipeline/stage3/tc_generator.py
# Converts Scenarios into ISO/IEC/IEEE 29119-3 compliant TestCase objects.

from dataclasses import dataclass, field
import re
from pipeline.stage1.rule_assembler import FormalRule, TestStrategy
from pipeline.stage2.scenario_generator import Scenario
from pipeline.stage2.knowledge_base import KnowledgeBase

_tc_counter = 0


def _next_tc_id() -> str:
    global _tc_counter
    _tc_counter += 1
    return f"TC-HEALTH-{_tc_counter:03d}"


@dataclass
class ISO29119TestCase:
    tc_id: str
    purpose: str
    priority: str
    classification: str
    preconditions: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    expected_result: str = ""
    suspension_criteria: str = ""
    postconditions: str = ""
    source_requirement: str = ""
    rule_id: str = ""
    scenario_id: str = ""
    strategy: str = ""


# ── Step Inference ───────────────────────────────────────────────────────────

def _infer_steps(action: str, scenario: Scenario, rule: FormalRule) -> list:
    role = scenario.inputs.get("user_role", rule.user_role or "clinician")
    param = next(iter([k for k in scenario.inputs if k not in ("user_role", "workflow",
                                                                 "from_state", "to_state")]), None)
    value = scenario.inputs.get(param, "threshold value") if param else "threshold value"
    workflow = scenario.inputs.get("workflow", "")
    from_state = scenario.inputs.get("from_state", "")
    to_state = scenario.inputs.get("to_state", "")

    if scenario.strategy == TestStrategy.STATE_TRANSITION:
        return [
            f"1. Login to the system as {role}.",
            f"2. Navigate to the {workflow.replace('_', ' ')} management module.",
            f"3. Locate or create a record in state '{from_state}'.",
            f"4. Attempt to transition the record to state '{to_state}'.",
            f"5. Observe the system response and verify the expected outcome.",
            f"6. Check audit logs to confirm the transition attempt was recorded.",
        ]

    if "alert" in action or "notify" in action or "send" in action or "escalate" in action:
        time_note = f" Wait for up to {rule.time_constraint}." if rule.time_constraint else ""
        return [
            f"1. Login to the system as {role}.",
            f"2. Navigate to the vital signs monitoring dashboard.",
            f"3. Set {param if param else 'the clinical parameter'} to {value}.",
            f"4. Wait for the system to process the reading (up to 5 seconds).{time_note}",
            f"5. Observe the alert panel for the expected notification.",
            f"6. Verify the alert content, severity, and recipient are correct.",
        ]

    if "restrict" in action or "prevent" in action or "access" in action.lower():
        return [
            f"1. Login to the system as {role}.",
            f"2. Attempt to access or modify the restricted resource/action.",
            f"3. Observe the system response.",
            f"4. Verify that access is denied and an appropriate error message is shown.",
            f"5. Confirm no unauthorized data was exposed or modified.",
        ]

    if "log" in action or "record" in action:
        return [
            f"1. Login to the system as {role}.",
            f"2. Perform the action that should trigger audit logging.",
            f"3. Navigate to the audit log module.",
            f"4. Search for the log entry corresponding to the performed action.",
            f"5. Verify that a timestamp and user ID are recorded in the log entry.",
        ]

    if "require" in action:
        return [
            f"1. Navigate to the system login page.",
            f"2. Enter valid primary credentials for a {role} account.",
            f"3. Observe whether a second authentication factor is requested.",
            f"4. Complete or decline the second factor as required by the test case.",
            f"5. Verify the expected access grant or denial.",
        ]

    if "display" in action:
        return [
            f"1. Login to the system as {role}.",
            f"2. Trigger the condition that requires the system to display a warning/alert.",
            f"3. Observe the user interface for the expected display element.",
            f"4. Verify the content, positioning, and timing of the displayed element.",
        ]

    if scenario.strategy == TestStrategy.TEMPORAL:
        time_val = scenario.inputs.get("response_time_measured", "N/A")
        unit     = scenario.inputs.get("unit", "")
        deadline = rule.time_constraint or "specified deadline"
        compliant = "non_compliant" not in scenario.expected_result.lower()
        verdict   = "is within" if compliant else "exceeds"
        return [
            f"1. Configure the monitoring infrastructure to record response timestamps.",
            f"2. Login to the system as {role}.",
            f"3. Trigger the condition that initiates the '{action}' process.",
            f"4. Record the exact start timestamp when the triggering event occurs.",
            f"5. Simulate that the measured response time is: {time_val} {unit}.",
            f"6. Compare measured response time against the required deadline: {deadline}.",
            f"7. Verify the response {verdict} the deadline — expected: {scenario.expected_result.replace('_', ' ')}.",
        ]

    if scenario.strategy == TestStrategy.CLINICAL_VALIDATION:
        param_key = next((k for k in scenario.inputs if k not in ("user_role",)), None)
        val       = scenario.inputs.get(param_key, "test value") if param_key else "test value"
        return [
            f"1. Login to the system as {role}.",
            f"2. Navigate to the data-entry interface for '{param_key or 'clinical parameter'}'.",
            f"3. Enter the value '{val}' for '{param_key or 'clinical parameter'}'.",
            f"4. Attempt to submit or save the entry.",
            f"5. Observe the system validation response.",
            f"6. Verify: expected outcome is '{scenario.expected_result.replace('_', ' ')}'.",
            f"7. Confirm no invalid data is persisted and an appropriate error message is displayed.",
        ]

    # Generic fallback
    return [
        f"1. Login to the system as {role}.",
        f"2. Navigate to the relevant module.",
        f"3. Set input parameters: {', '.join(f'{k}={v}' for k, v in scenario.inputs.items())}.",
        f"4. Trigger the action: {action}.",
        f"5. Observe and record the system response.",
        f"6. Compare actual result with expected result.",
    ]


def _infer_preconditions(rule: FormalRule, scenario: Scenario) -> list:
    preconditions = ["Patient record exists in the system."]

    param = next(iter([k for k in scenario.inputs if k not in ("user_role", "workflow",
                                                                 "from_state", "to_state")]), None)
    if param:
        preconditions.append(f"{param} sensor or input mechanism is connected and operational.")

    role = scenario.inputs.get("user_role", rule.user_role)
    if role:
        preconditions.append(f"A user account with role '{role}' exists and is active.")

    if scenario.strategy == TestStrategy.STATE_TRANSITION:
        wf = scenario.inputs.get("workflow", "workflow")
        from_state = scenario.inputs.get("from_state", "initial state")
        preconditions.append(f"A {wf.replace('_', ' ')} record exists in state '{from_state}'.")
        preconditions.append("The user has appropriate permissions for the workflow.")

    if scenario.strategy == TestStrategy.TEMPORAL:
        preconditions.append("System clock is synchronized and accurate to within 0.1 seconds.")
        preconditions.append("Response-time monitoring infrastructure is in place and calibrated.")
    elif rule.time_constraint:
        preconditions.append("System clock is synchronized and accurate.")

    preconditions.append("System is running and all services are healthy.")
    return preconditions


def _infer_suspension_criteria(rule: FormalRule, scenario: Scenario) -> str:
    param = next(iter([k for k in scenario.inputs if k not in ("user_role", "workflow",
                                                                 "from_state", "to_state")]), None)
    if param:
        return f"System unavailable, {param} sensor hardware fault detected, or test environment is unresponsive."
    if scenario.strategy == TestStrategy.STATE_TRANSITION:
        return "System unavailable, database connection lost, or workflow engine is unresponsive."
    return "System unavailable or test environment is unresponsive."


def _infer_postconditions(rule: FormalRule, scenario: Scenario) -> str:
    result = scenario.expected_result.lower()
    if "alert_triggered" in result:
        return "Alert remains in TRIGGERED state until acknowledged by an authorized user."
    if "access_denied" in result:
        return "No patient data was exposed. Access attempt is recorded in the audit log."
    if "audit_log" in result:
        return "The audit log entry is permanent and includes timestamp and user ID."
    if "transition_success" in result:
        return "The workflow record is now in the new state. Transition is logged."
    if "transition_rejected" in result:
        return "The workflow record remains in its original state. Rejection is logged."
    if "warning_displayed" in result:
        return "Warning remains visible until dismissed by an authorized user."
    if "authentication_required" in result:
        return "User session is not created until all required authentication factors are completed."
    if "non_compliant_timeout" in result:
        return "An alert or escalation event has been triggered due to the deadline breach. Incident is logged with timestamp."
    if "compliant" in result and "boundary" not in result:
        return "Response was delivered within the required deadline. No escalation triggered."
    if "boundary_" in result:
        return "Response arrived exactly at the deadline boundary. System behavior at edge case is recorded."
    if "validation_error" in result:
        return "Invalid input was rejected. No data was persisted. Validation error is logged with details."
    if "accepted_no_alert" in result:
        return "Valid reading accepted. No alert triggered. System state is unchanged."
    return "System returns to a stable state. All actions are recorded in the audit trail."


def _classify(strategy: TestStrategy, scenario_type: str) -> str:
    if strategy == TestStrategy.BVA:
        return f"boundary/functional ({scenario_type})"
    if strategy == TestStrategy.EP:
        return f"equivalence/access-control ({scenario_type})"
    if strategy == TestStrategy.STATE_TRANSITION:
        return f"state-transition/workflow ({scenario_type})"
    if strategy == TestStrategy.DECISION_TABLE:
        return f"decision-table/multi-condition ({scenario_type})"
    if strategy == TestStrategy.TEMPORAL:
        return f"temporal/timing-deadline ({scenario_type})"
    if strategy == TestStrategy.CLINICAL_VALIDATION:
        return f"clinical-validation/input-safety ({scenario_type})"
    return f"functional ({scenario_type})"


def _infer_purpose(scenario: Scenario, rule: FormalRule) -> str:
    action = rule.action
    param = next((k for k in scenario.inputs if k not in ("user_role", "workflow",
                                                            "from_state", "to_state")), None)
    value = scenario.inputs.get(param) if param else None
    role = scenario.inputs.get("user_role", rule.user_role)
    from_s = scenario.inputs.get("from_state")
    to_s = scenario.inputs.get("to_state")

    if scenario.strategy == TestStrategy.BVA and param and value is not None:
        return (f"Verify system {action}s when {param} = {value} "
                f"({scenario.scenario_type.replace('_', ' ')})")
    if scenario.strategy == TestStrategy.EP and role:
        return f"Verify {action} behavior for user role '{role}' ({scenario.scenario_type.replace('_', ' ')})"
    if scenario.strategy == TestStrategy.STATE_TRANSITION and from_s and to_s:
        return f"Verify {scenario.inputs.get('workflow', 'workflow')} transition {from_s} → {to_s} ({scenario.scenario_type.replace('_', ' ')})"
    if scenario.strategy == TestStrategy.DECISION_TABLE:
        return f"Verify {action} outcome for condition combination: {scenario.scenario_type.replace('_', ' ')}"
    if scenario.strategy == TestStrategy.TEMPORAL:
        return (f"Verify '{action}' meets timing deadline — "
                f"{scenario.scenario_type.replace('_', ' ')} "
                f"(measured: {scenario.inputs.get('response_time_measured', '?')} "
                f"{scenario.inputs.get('unit', '')})")
    if scenario.strategy == TestStrategy.CLINICAL_VALIDATION:
        cv_param = next((k for k in scenario.inputs if k not in ("user_role",)), param)
        return (f"Verify clinical input validation for '{cv_param}': "
                f"{scenario.scenario_type.replace('_', ' ')}")
    return f"Verify system behavior: {action} — {scenario.scenario_type.replace('_', ' ')}"


def _infer_dependencies(rule: FormalRule, scenario: Scenario, all_rules: list) -> list:
    deps = []
    # State transitions depend on earlier workflow test cases
    if scenario.strategy == TestStrategy.STATE_TRANSITION:
        for r in all_rules:
            if r.rule_id != rule.rule_id and r.strategies and TestStrategy.STATE_TRANSITION in r.strategies:
                deps.append(r.rule_id.replace("RULE", "TC-HEALTH"))
                break
    return deps


# ── Cartesian Product expansion ──────────────────────────────────────────────

def _expand_inputs(scenario: Scenario, kb: KnowledgeBase) -> list:
    """
    If any input value is a list, expand via Cartesian product.
    Returns a list of concrete input dicts.
    """
    from itertools import product

    list_keys = [k for k, v in scenario.inputs.items() if isinstance(v, list)]
    if not list_keys:
        return [dict(scenario.inputs)]

    fixed = {k: v for k, v in scenario.inputs.items() if k not in list_keys}
    list_values = [scenario.inputs[k] for k in list_keys]

    expanded = []
    for combo in product(*list_values):
        inp = dict(fixed)
        for k, v in zip(list_keys, combo):
            inp[k] = v
        expanded.append(inp)
    return expanded


# ── Main Entry Point ─────────────────────────────────────────────────────────

def generate_test_cases(scenarios: list, rules: list, kb) -> list:
    """
    Generate ISO29119 test cases from scenarios + formal rules.

    Args:
        scenarios: List of Scenario objects.
        rules: List of FormalRule objects.
        kb: KnowledgeBase or SessionKB.

    Returns:
        List of ISO29119TestCase objects.
    """
    global _tc_counter
    _tc_counter = 0

    rule_map = {r.rule_id: r for r in rules}
    test_cases = []

    for scenario in scenarios:
        rule = rule_map.get(scenario.rule_id)
        if not rule:
            continue

        # Expand Cartesian product for multi-value inputs
        expanded_inputs = _expand_inputs(scenario, kb)

        for inp in expanded_inputs:
            # Merge expanded inputs back into a scenario-like object for step inference
            class _PseudoScenario:
                pass
            ps = _PseudoScenario()
            ps.inputs = inp
            ps.strategy = scenario.strategy
            ps.scenario_type = scenario.scenario_type
            ps.expected_result = scenario.expected_result

            steps = _infer_steps(rule.action, ps, rule)
            preconditions = _infer_preconditions(rule, ps)
            suspension = _infer_suspension_criteria(rule, ps)
            postconditions = _infer_postconditions(rule, ps)
            classification = _classify(scenario.strategy, scenario.scenario_type)
            purpose = _infer_purpose(ps, rule)
            dependencies = _infer_dependencies(rule, ps, rules)

            # Build expected result string
            expected_str = scenario.expected_result.replace("_", " ").replace("no ", "No ")
            if rule.time_constraint:
                # Strip leading "within" / "after" / "in" already present in the stored string
                tc_str = str(rule.time_constraint).strip()
                tc_str = re.sub(r'^(?:within|after|in)\s+', '', tc_str, flags=re.IGNORECASE)
                expected_str += f" (within {tc_str})"

            tc = ISO29119TestCase(
                tc_id=_next_tc_id(),
                purpose=purpose,
                priority=scenario.priority,
                classification=classification,
                preconditions=preconditions,
                dependencies=dependencies,
                inputs=inp,
                steps=steps,
                expected_result=expected_str,
                suspension_criteria=suspension,
                postconditions=postconditions,
                source_requirement=rule.source_sentence,
                rule_id=rule.rule_id,
                scenario_id=scenario.scenario_id,
                strategy=scenario.strategy.value,
            )
            test_cases.append(tc)

    return test_cases


def _to_dict(tc: ISO29119TestCase) -> dict:
    """Convert an ISO29119TestCase to a JSON-serializable dict."""
    return {
        "tc_id": tc.tc_id,
        "purpose": tc.purpose,
        "priority": tc.priority,
        "classification": tc.classification,
        "preconditions": tc.preconditions,
        "dependencies": tc.dependencies,
        "inputs": tc.inputs,
        "steps": tc.steps,
        "expected_result": tc.expected_result,
        "suspension_criteria": tc.suspension_criteria,
        "postconditions": tc.postconditions,
        "source_requirement": tc.source_requirement,
        "rule_id": tc.rule_id,
        "scenario_id": tc.scenario_id,
        "strategy": tc.strategy,
    }
