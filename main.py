#!/usr/bin/env python3
# main.py — TestGenX v3.0
# Architecture: Stage0 → SessionKB → Stage1 → Stage2 → Stage3
# No static fallback. No unresolved values. No inflated TC count.

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import LINGUISTIC_REGISTRY_PATH, OUTPUT_DIR, GEMINI_API_KEY, GEMINI_MODEL
from pipeline.ingestion import load_document
from pipeline.session_kb import SessionKB
from pipeline.gemini_srs_analyzer import GeminiSRSAnalyzer
from pipeline.stage0.srs_bootstrap import build_session_kb
from pipeline.stage1.testability_filter import classify_sentences
from pipeline.stage1.ner_extractor import extract_entities
from pipeline.stage1.rule_assembler import assemble_rules
from pipeline.stage2.scenario_generator import generate_scenarios
from pipeline.stage3.tc_generator import generate_test_cases, _to_dict
from pipeline.stage3.augmentor import augment_with_edge_cases


def main():
    parser = argparse.ArgumentParser(
        description="TestGenX v3.0 — Automated Healthcare Test Case Generator"
    )
    parser.add_argument("--input",   required=True,
                        help="Path to SRS document (PDF, DOCX, or TXT)")
    parser.add_argument("--output",  default="output/test_cases.json")
    parser.add_argument("--augment", action="store_true",
                        help="Enable GPT-4o edge case augmentation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print intermediate results")
    args = parser.parse_args()

    print("=" * 65)
    print("  TestGenX v3.0 — Dynamic Healthcare Test Case Generator")
    print("  ISO/IEC/IEEE 29119-3 | No static KB | No unresolved TCs")
    print("=" * 65)
    start = time.time()

    # ── INIT SessionKB ────────────────────────────────────────────────────────
    print(f"\n[INIT] Loading linguistic registry...")
    skb = SessionKB(LINGUISTIC_REGISTRY_PATH)

    # ── INGEST ────────────────────────────────────────────────────────────────
    print(f"\n[INGEST] Reading document: {args.input}")
    try:
        sentences = load_document(args.input)
    except FileNotFoundError as e:
        print(f"\n[ERR] {e}")
        sys.exit(1)

    skb.all_sentences = sentences
    print(f"  -> {len(sentences)} sentences loaded")

    # ── GEMINI: Single API call over the whole SRS ────────────────────────────────────
    print(f"\n[GEMINI] Analysing SRS with {GEMINI_MODEL}...")
    analyzer = GeminiSRSAnalyzer(model_name=GEMINI_MODEL, api_key=GEMINI_API_KEY)
    analyzer.analyze(sentences)   # result cached on analyzer; re-used by all stages

    # ── STAGE 0: SRS Bootstrap ───────────────────────────────────────────────────────────
    print(f"\n[STAGE 0] SRS Bootstrap — populating SessionKB from Gemini analysis...")
    build_session_kb(sentences, skb, analyzer, verbose=args.verbose)

    print(f"\n  SessionKB populated:")
    s = skb.summary()
    print(f"    [OK] {s['params_resolved']} clinical params resolved")
    print(f"    [OK] {s['roles_extracted']} roles extracted from SRS")
    print(f"    [OK] {s['workflows_extracted']} workflows extracted from SRS")
    if s['params_skipped'] > 0:
        print(f"    [WARN] {s['params_skipped']} params unresolvable -> rules using them will be skipped")
        for sp in skb.skipped_params:
            print(f"       - {sp['param']}: {sp['reason']}")

    # ── STAGE 1: Classify → NER → Rules ──────────────────────────────────────
    print(f"\n[STAGE 1] Rule Extraction")

    classified = classify_sentences(sentences, analyzer)
    skb.classified_sentences = classified

    testable         = [s for s in classified if s.label == "TESTABLE"]
    domain_knowledge = [s for s in classified if s.label == "DOMAIN_KNOWLEDGE"]
    not_testable     = [s for s in classified if s.label == "NOT_TESTABLE"]

    print(f"  -> {len(testable)} TESTABLE | "
          f"{len(domain_knowledge)} DOMAIN_KNOWLEDGE | "
          f"{len(not_testable)} NOT_TESTABLE")

    if args.verbose:
        icons = {"TESTABLE": "[T]", "DOMAIN_KNOWLEDGE": "[D]", "NOT_TESTABLE": "[N]"}
        for cs in classified:
            print(f"    {icons[cs.label]} [{cs.label:<20}] ({cs.confidence:.2f}) {cs.text[:70]}")

    ner_results = extract_entities([s.text for s in testable], analyzer)
    skb.ner_results = ner_results

    if args.verbose:
        print("\n  NER Results:")
        for nr in ner_results:
            print(f"    * {nr.sentence[:65]}")
            for etype, vals in nr.entities.items():
                print(f"        [{etype}]: {vals}")

    rules = assemble_rules(ner_results, skb)
    skb.formal_rules = rules
    print(f"  -> {len(rules)} rules assembled")

    if skb.skipped_rules:
        print(f"  [WARN] {len(skb.skipped_rules)} rules skipped (unresolvable params):")
        for sr in skb.skipped_rules:
            print(f"     - {sr['rule_id']}: {sr['reason']}")
            print(f"       Sentence: {sr['sentence'][:70]}")

    if args.verbose:
        for r in rules:
            strats = [s.value for s in r.strategies]
            print(f"    {r.rule_id}: action={r.action} | "
                  f"conds={len(r.conditions)} | role={r.user_role} | strategies={strats}")

    # ── STAGE 2: Scenario Generation ─────────────────────────────────────────
    print(f"\n[STAGE 2] Scenario Generation")
    scenarios = generate_scenarios(rules, skb)
    skb.scenarios = scenarios
    print(f"  -> {len(scenarios)} scenarios generated")

    if args.verbose:
        for sc in scenarios:
            print(f"    {sc.scenario_id}: {sc.strategy.value} | "
                  f"{sc.scenario_type} | {sc.priority} | inputs={sc.inputs}")

    # ── STAGE 3: Test Case Generation ────────────────────────────────────────
    print(f"\n[STAGE 3] Test Case Generation")
    test_cases = generate_test_cases(scenarios, rules, skb)
    skb.test_cases = test_cases
    print(f"  -> {len(test_cases)} test cases generated")

    if args.augment:
        test_cases = augment_with_edge_cases(test_cases)
        print(f"  -> After augmentation: {len(test_cases)} test cases")

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = round(time.time() - start, 2)

    strategy_counts: dict = {}
    priority_counts: dict = {}
    source_counts: dict   = {}
    for tc in test_cases:
        strategy_counts[tc.strategy]  = strategy_counts.get(tc.strategy, 0) + 1
        priority_counts[tc.priority]  = priority_counts.get(tc.priority, 0) + 1

    # Track knowledge source breakdown across all TC inputs
    for rule in rules:
        for cond in rule.conditions:
            src = cond.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

    output = {
        "pipeline":              "TestGenX v3.0",
        "standard":              "ISO/IEC/IEEE 29119-3",
        "input_document":        args.input,
        "total_sentences":       len(sentences),
        "total_testable":        len(testable),
        "total_rules":           len(rules),
        "rules_skipped":         len(skb.skipped_rules),
        "params_skipped":        skb.skipped_params,
        "total_scenarios":       len(scenarios),
        "total_test_cases":      len(test_cases),
        "strategy_breakdown":    strategy_counts,
        "priority_breakdown":    priority_counts,
        "knowledge_source_breakdown": source_counts,
        "session_kb_summary":    skb.summary(),
        "elapsed_seconds":       elapsed,
        "test_cases":            [_to_dict(tc) for tc in test_cases],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Plain-text report ─────────────────────────────────────────────────────
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  TestGenX v3.0 — ISO/IEC/IEEE 29119-3 Test Case Report\n")
        f.write(f"  Input   : {args.input}\n")
        f.write(f"  Total   : {len(test_cases)} test cases\n")
        f.write(f"  Elapsed : {elapsed}s\n")
        f.write("=" * 70 + "\n\n")

        for tc_dict in output["test_cases"]:
            f.write("-" * 70 + "\n")
            f.write(f"ID          : {tc_dict['tc_id']}\n")
            f.write(f"Purpose     : {tc_dict['purpose']}\n")
            f.write(f"Strategy    : {tc_dict['strategy']}\n")
            f.write(f"Priority    : {tc_dict['priority']}\n")
            f.write(f"Class       : {tc_dict['classification']}\n")
            f.write(f"Rule        : {tc_dict['rule_id']}\n")
            if tc_dict.get('source_requirement'):
                f.write(f"Requirement : {tc_dict['source_requirement']}\n")
            f.write("\nPreconditions:\n")
            for pre in tc_dict.get('preconditions', []):
                f.write(f"  - {pre}\n")
            f.write("\nInputs:\n")
            for k, v in tc_dict.get('inputs', {}).items():
                f.write(f"  {k}: {v}\n")
            f.write("\nSteps:\n")
            for step in tc_dict.get('steps', []):
                f.write(f"  {step}\n")
            f.write(f"\nExpected Result : {tc_dict['expected_result']}\n")
            f.write(f"Postconditions  : {tc_dict.get('postconditions', '')}\n")
            if tc_dict.get('suspension_criteria'):
                f.write(f"Suspend If      : {tc_dict['suspension_criteria']}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"  Strategy breakdown : {strategy_counts}\n")
        f.write(f"  Priority breakdown : {priority_counts}\n")
        f.write("=" * 70 + "\n")

    print(f"\n{'=' * 65}")
    print(f"  [OK] Done in {elapsed}s")
    print(f"  Output (JSON) : {output_path}")
    print(f"  Output (TXT)  : {txt_path}")
    print(f"  Test cases    : {len(test_cases)}")
    print(f"  Strategies    : {strategy_counts}")
    print(f"  Priorities    : {priority_counts}")
    print(f"  Knowledge src : {source_counts}")
    if skb.skipped_rules:
        print(f"  [WARN] Rules skipped: {len(skb.skipped_rules)} (unresolvable params)")
    print("=" * 65)


if __name__ == "__main__":
    main()