# TestGenX — Automated Healthcare Test Case Generator

> Reads a Healthcare Software Requirements Specification (SRS) and produces structured, **ISO/IEC/IEEE 29119-3** compliant test cases using a 3-stage AI pipeline.

---

## Architecture

```
SRS Document (PDF/DOCX/TXT)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Rule Extraction                                  │
│  ├── Ingestion (spaCy sentence segmentation)                │
│  ├── Testability Filter (facebook/bart-large-mnli)          │
│  │     → TESTABLE / DOMAIN_KNOWLEDGE / NOT_TESTABLE         │
│  ├── NER Extractor (d4data/biomedical-ner-all + regex)      │
│  │     → ACTION, USER_ROLE, CLINICAL_PARAM, OPERATOR, ...   │
│  └── Rule Assembler → FormalRule objects                    │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — Scenario Generation (Pure Algorithm, no ML)      │
│  ├── Knowledge Base (clinical params, roles, workflows)     │
│  └── Scenario Generator                                     │
│        ├── BVA  — Boundary Value Analysis                   │
│        ├── EP   — Equivalence Partitioning                  │
│        ├── ST   — State Transition Testing                  │
│        └── DT   — Decision Table Testing                    │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — Test Case Generation                             │
│  ├── TC Generator (ISO 29119-3 format, Cartesian expansion) │
│  └── Augmentor (optional GPT-4o edge cases)                 │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
  output/test_cases.json
```

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download spaCy language model
python -m spacy download en_core_web_sm
```

> **Note:** On first run, `facebook/bart-large-mnli` (~1.6 GB) and `d4data/biomedical-ner-all` will download from HuggingFace automatically and are cached locally.

---

## Run on the sample SRS

```bash
python main.py --input data/sample_srs/sample_srs.txt
```

With verbose output:

```bash
python main.py --input data/sample_srs/sample_srs.txt --verbose
```

Custom output path:

```bash
python main.py --input data/sample_srs/sample_srs.txt --output output/my_test_cases.json
```

---

## Run on your own SRS document

```bash
# PDF
python main.py --input path/to/your/srs.pdf

# Word document
python main.py --input path/to/your/srs.docx

# Plain text
python main.py --input path/to/your/srs.txt
```

---

## Enable GPT-4o edge case augmentation (optional)

```bash
export OPENAI_API_KEY=sk-...
python main.py --input data/sample_srs/sample_srs.txt --augment
```

Augmentation adds up to 5 edge cases per test case (configurable in `config.py`).

---

## Output Format

Results are saved to `output/test_cases.json` — fully populated ISO/IEC/IEEE 29119-3 test cases:

```json
{
  "pipeline": "TestGenX v1.0",
  "standard": "ISO/IEC/IEEE 29119-3",
  "total_test_cases": 142,
  "strategy_breakdown": { "BVA": 50, "EP": 40, "STATE_TRANSITION": 30, "DECISION_TABLE": 22 },
  "test_cases": [
    {
      "tc_id": "TC-HEALTH-001",
      "purpose": "Verify system alerts when SpO2 = 89 (boundary below)",
      "priority": "HIGH",
      "classification": "boundary/functional (boundary_below)",
      "preconditions": ["Patient record exists in the system.", "SpO2 sensor connected."],
      "inputs": { "SpO2": 89.0, "user_role": "nurse" },
      "steps": ["1. Login as nurse.", "2. Navigate to monitoring dashboard.", "..."],
      "expected_result": "alert triggered (within 5 seconds)",
      "suspension_criteria": "System unavailable or sensor hardware fault detected.",
      "postconditions": "Alert remains in TRIGGERED state until acknowledged.",
      "source_requirement": "The system shall alert the nurse when SpO2 drops below 90%.",
      "rule_id": "RULE-001",
      "strategy": "BVA"
    }
  ]
}
```

---

## Configuration

Edit `config.py` to tune pipeline behaviour:

| Setting | Default | Description |
|---|---|---|
| `TESTABILITY_THRESHOLD` | `0.4` | Minimum confidence to classify a sentence as TESTABLE |
| `BVA_WELL_INSIDE_OFFSET` | `5` | Steps from boundary for well-inside BVA values |
| `EDGE_CASES_PER_TC` | `5` | GPT-4o edge cases generated per test case |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model for augmentation |

---

## Project Structure

```
testgenx/
├── main.py                          # CLI entry point
├── config.py                        # All tunable settings
├── requirements.txt
├── data/
│   ├── knowledge_base/knowledge.json  # Clinical KB (parameters, roles, workflows)
│   └── sample_srs/sample_srs.txt     # Sample healthcare SRS
├── pipeline/
│   ├── ingestion.py                 # Document reader + sentence segmentation
│   ├── stage1/
│   │   ├── testability_filter.py    # Zero-shot classification (BART)
│   │   ├── ner_extractor.py         # Biomedical NER + regex fallback
│   │   └── rule_assembler.py        # FormalRule construction + strategy tagging
│   ├── stage2/
│   │   ├── knowledge_base.py        # KB loader + lookup methods
│   │   └── scenario_generator.py   # BVA / EP / ST / DT algorithms
│   └── stage3/
│       ├── tc_generator.py          # ISO 29119-3 test case formatting
│       └── augmentor.py             # Optional GPT-4o edge case generation
└── output/
    └── test_cases.json              # Generated test cases
```
