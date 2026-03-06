import json
from pathlib import Path

OUTPUT_JSON = "output/test_cases.json"
OUTPUT_TXT  = "output/test_cases.txt"

with open(OUTPUT_JSON, encoding="utf-8") as f:
    data = json.load(f)

lines = []
lines.append("=" * 70)
lines.append("  TestGenX v3.0 — Generated Test Cases")
lines.append(f"  Standard  : ISO/IEC/IEEE 29119-3")
lines.append(f"  Input     : {data['input_document']}")
lines.append(f"  Total TCs : {data['total_test_cases']}")
lines.append(f"  Rules     : {data['total_rules']} assembled, {data['rules_skipped']} skipped")
lines.append(f"  Strategies: {data['strategy_breakdown']}")
lines.append(f"  Priorities: {data['priority_breakdown']}")
lines.append(f"  Elapsed   : {data['elapsed_seconds']}s")
lines.append("=" * 70)
lines.append("")

for tc in data["test_cases"]:
    tc_id  = tc.get("tc_id") or tc.get("id", "?")
    title  = tc.get("purpose") or tc.get("title") or tc.get("name", "")
    lines.append(f"{'─' * 70}")
    lines.append(f"[{tc_id}]  Priority: {tc.get('priority','?')}  |  Strategy: {tc.get('strategy','')}  |  {tc.get('classification','')}")
    lines.append(f"Purpose   : {title}")
    lines.append(f"Source    : {tc.get('source_requirement','')[:80]}")
    lines.append("Preconditions:")
    for p in tc.get("preconditions", []):
        lines.append(f"  - {p}")
    lines.append(f"Inputs    : {tc.get('inputs') or tc.get('test_inputs','')}")
    lines.append("Steps:")
    for s in tc.get("steps", []):
        lines.append(f"  {s}")
    lines.append(f"Expected  : {tc.get('expected_result','')}")
    lines.append(f"Post-cond : {tc.get('postconditions','')}")
    if tc.get("suspension_criteria"):
        lines.append(f"Suspend if: {tc['suspension_criteria']}")
    lines.append("")

lines.append("=" * 70)
lines.append(f"END — {data['total_test_cases']} test cases")
lines.append("=" * 70)

full_text = "\n".join(lines)

# Print to console
print(full_text)

# Save to file
Path(OUTPUT_TXT).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_TXT).write_text(full_text, encoding="utf-8")
print(f"\n[Saved] {OUTPUT_TXT}")

