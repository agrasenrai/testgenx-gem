# pipeline/stage3/augmentor.py
# Optional GPT-4o edge case augmentation. Silently skips if no OPENAI_API_KEY.

import os
import json
import logging
from pipeline.stage3.tc_generator import ISO29119TestCase, _to_dict
from config import OPENAI_MODEL, EDGE_CASES_PER_TC

logger = logging.getLogger(__name__)

_EDGE_CASE_PROMPT_TEMPLATE = """You are a senior healthcare software QA engineer.

Given the following test case for an ISO/IEC/IEEE 29119-3 compliant healthcare system:

Test Case ID: {tc_id}
Purpose: {purpose}
Inputs: {inputs}
Expected Result: {expected_result}
Source Requirement: {source_requirement}

Generate exactly {n} additional edge case test cases that cover different failure modes.
Focus on: security attacks, concurrency issues, integration failures, error recovery, and performance limits.

Respond ONLY with a JSON array of test case objects. Each object must have these exact fields:
- purpose (string)
- inputs (object with key-value pairs)
- expected_result (string)
- steps (array of strings)
- preconditions (array of strings)
- suspension_criteria (string)
- postconditions (string)
- priority (one of: HIGH, MEDIUM, LOW)

Do not include any preamble, markdown, or explanation. Only the JSON array."""


def _parse_edge_cases(raw_text: str, base_tc: ISO29119TestCase, start_idx: int) -> list:
    """Parse GPT JSON response into ISO29119TestCase objects."""
    text = raw_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        edge_cases_data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse GPT edge cases JSON: {e}")
        return []

    if not isinstance(edge_cases_data, list):
        logger.warning("GPT response was not a JSON array.")
        return []

    results = []
    for i, ec_data in enumerate(edge_cases_data):
        if not isinstance(ec_data, dict):
            continue
        tc = ISO29119TestCase(
            tc_id=f"{base_tc.tc_id}-EC-{start_idx + i + 1}",
            purpose=ec_data.get("purpose", f"Edge case {i+1} for {base_tc.tc_id}"),
            priority=ec_data.get("priority", "MEDIUM"),
            classification=f"edge-case/{base_tc.strategy.lower()}",
            preconditions=ec_data.get("preconditions", base_tc.preconditions[:]),
            dependencies=[base_tc.tc_id],
            inputs=ec_data.get("inputs", dict(base_tc.inputs)),
            steps=ec_data.get("steps", ["Perform edge case test as described."]),
            expected_result=ec_data.get("expected_result", "System handles edge case gracefully."),
            suspension_criteria=ec_data.get("suspension_criteria", base_tc.suspension_criteria),
            postconditions=ec_data.get("postconditions", base_tc.postconditions),
            source_requirement=base_tc.source_requirement,
            rule_id=base_tc.rule_id,
            scenario_id=f"{base_tc.scenario_id}-EC",
            strategy=base_tc.strategy,
        )
        results.append(tc)

    return results


def augment_with_edge_cases(
    test_cases: list,
    max_augment: int = 5,
) -> list:
    """
    Augment test cases with GPT-4o generated edge cases.
    Silently skips if no OPENAI_API_KEY is found or if any API call fails.

    Args:
        test_cases: List of ISO29119TestCase objects.
        max_augment: Only augment the first N test cases (cost control).

    Returns:
        Extended list of ISO29119TestCase objects (originals + edge cases).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Augmentor] No OPENAI_API_KEY found. Skipping edge case augmentation.")
        return test_cases

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("[Augmentor] openai package not installed. Skipping augmentation.")
        return test_cases

    augmented = list(test_cases)
    tcs_to_augment = test_cases[:max_augment]

    print(f"[Augmentor] Augmenting {len(tcs_to_augment)} test cases with GPT-4o edge cases...")

    for base_tc in tcs_to_augment:
        try:
            prompt = _EDGE_CASE_PROMPT_TEMPLATE.format(
                tc_id=base_tc.tc_id,
                purpose=base_tc.purpose,
                inputs=json.dumps(base_tc.inputs),
                expected_result=base_tc.expected_result,
                source_requirement=base_tc.source_requirement,
                n=EDGE_CASES_PER_TC,
            )

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )

            raw_text = response.choices[0].message.content
            edge_cases = _parse_edge_cases(raw_text, base_tc, start_idx=0)
            augmented.extend(edge_cases)
            print(f"  [Augmentor] Added {len(edge_cases)} edge cases for {base_tc.tc_id}")

        except Exception as e:
            logger.warning(f"[Augmentor] Failed to augment {base_tc.tc_id}: {e}. Skipping.")
            continue

    print(f"[Augmentor] Augmentation complete. Total test cases: {len(augmented)}")
    return augmented
