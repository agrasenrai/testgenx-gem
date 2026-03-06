# TestGenX v3.0 — Full Project Analysis (Updated March 2026)

---

## 1. What the Project Does

TestGenX is an **automated test case generator for healthcare software**. Given a plain-text,
PDF, or DOCX Software Requirements Specification (SRS) document, it reads the requirements and
outputs a structured JSON file of test cases that fully conform to **ISO/IEC/IEEE 29119-3** —
the international standard for software test documentation.

**The goal:** A QA engineer drops in an SRS file and gets professional, prioritised, traceable
test cases out — zero manual writing, zero hardcoded knowledge.

### Project Overview

TestGenX reads a healthcare SRS document, sends the full text to Google Gemini in a single API
call, and gets back two things: a structured knowledge base (clinical parameters with numeric
thresholds, user roles with access levels, workflow state machines, and time constraints) and a
per-sentence analysis (testability label, named entity extraction across 7 entity types, and a
Gemini-written expected result string). That knowledge base is loaded into a central SessionKB
object, and the per-sentence entity data is assembled into FormalRule objects — one per
testable requirement — each tagged with one or more test strategies (BVA for numeric
thresholds, EP for role-based access control, STATE_TRANSITION for workflow state machines,
DECISION_TABLE for multi-condition logic, TEMPORAL for timing/deadline requirements,
CLINICAL_VALIDATION for invalid/null/out-of-range input validation). A scenario generator
expands each rule into concrete test scenarios using the strategy algorithms, and a final
template engine turns every scenario into a fully-populated ISO 29119-3 test case with purpose,
preconditions, numbered steps, expected result, suspension criteria, and full traceability back
to the source SRS sentence. The end result is a JSON file and a human-readable text file, both
containing professional test cases that a QA team can execute directly — generated in under 90
seconds from a plain text requirements document.

---

### Verified Run Result (last successful run)

```
Input   : sample_srs.txt
Runtime : ~70 seconds
Outputs : output/test_cases.json
          output/test_cases.txt

Sentences processed : 32
Testable sentences  : 32  (100% — Gemini classifies all sentences)
Rules assembled     : 32  -> active rules produce scenarios
Test cases output   : 67

Strategies  : BVA=8   EP=33   CLINICAL_VALIDATION=10   TEMPORAL=16
Priorities  : HIGH=~28  MEDIUM=~35  LOW=~4
```

---

### Before / After Comparison

| Metric                       | Before (NLI + BioGPT + NER-ML)    | After (Gemini, v3.0)              |
|------------------------------|-----------------------------------|-----------------------------------|
| Testable sentences           | 8 / 32 (25%)                      | 32 / 32 (100%)                    |
| Active rules                 | 5                                 | 32                                |
| Test cases                   | 16                                | **67**                            |
| Test strategies              | BVA + EP + STATE_TRANSITION       | **6 strategies**                  |
| BVA test cases               | 0                                 | **8**                             |
| STATE_TRANSITION TCs         | 4                                 | active (KB-driven)                |
| TEMPORAL TCs                 | 0                                 | **16**                            |
| CLINICAL_VALIDATION TCs      | 0                                 | **10**                            |
| Drug interaction TCs         | 0                                 | **yes (event-based EP)**          |
| Models required              | transformers + torch + biogpt + ML | **google-generativeai only**     |
| Hardcoded domain words       | Yes (roles, workflow names, etc.) | **None — fully dynamic**          |
| Time-constraint formatting   | "within within 5 seconds" bug     | **Clean: (within 5 seconds)**     |
| BVA boundary_exact logic     | Wrong expected result for `<`      | **Fixed: no_{result} at boundary**|
| EP role selection            | Wrong role when rule role not in KB| **Fixed: rule.user_role direct** |

---

## 2. Project Structure

```
files/
+-- main.py                         Entry point — orchestrates full pipeline; saves JSON + TXT
+-- config.py                       All tunable settings + API key
+-- linguistic_registry             JSON — NLP operator/action/condition patterns (dynamic)
+-- sample_srs.txt                  Example SRS input (Healthcare EHR SRS)
+-- requirements.txt                Python dependencies
+-- README.md                       Project readme
+-- project_analysis.md             This file
+-- output/
|   +-- test_cases.json             Generated test cases (ISO 29119-3, machine-readable)
|   +-- test_cases.txt              Human-readable formatted test cases
+-- pipeline/
    +-- ingestion.py                Document reader + sentence splitter (spaCy)
    +-- session_kb.py               Central context object; state_transition_indicator_words
    |                               now fully dynamic (derived from workflow state names
    |                               and trigger_keywords — no hardcoded word list)
    +-- gemini_srs_analyzer.py      Gemini API wrapper — single-call classification + NER
    |
    +-- stage0/                     SRS Bootstrap — domain knowledge extraction
    |   +-- __init__.py
    |   +-- srs_bootstrap.py        Reads Gemini knowledge_base -> populates SessionKB
    |   +-- clinical_retriever.py   (Legacy — no longer called)
    |
    +-- stage1/                     Rule Extraction
    |   +-- __init__.py
    |   +-- testability_filter.py   Reads Gemini classification results
    |   +-- ner_extractor.py        Reads Gemini NER + expected_result; NERResult now
    |   |                           carries test_strategies: list from Gemini output
    |   +-- rule_assembler.py       NERResult -> FormalRule; _resolve_strategies() replaces
    |                               old heuristic _detect_strategies(); 6 TestStrategy enums
    |
    +-- stage2/                     Scenario Generation
    |   +-- __init__.py
    |   +-- scenario_generator.py   BVA / EP / STATE_TRANSITION / DECISION_TABLE /
    |   |                           TEMPORAL / CLINICAL_VALIDATION; no hardcoded roles,
    |   |                           no hardcoded workflow keywords; fully KB-driven
    |   +-- knowledge_base.py       Legacy KB class (type reference only)
    |
    +-- stage3/                     Test Case Generation
        +-- __init__.py
        +-- tc_generator.py         Scenario -> ISO29119TestCase; step templates for all
        |                           6 strategies; TEMPORAL + CLINICAL_VALIDATION added
        +-- augmentor.py            Optional GPT-4o edge case augmentation
```

---

## 3. Full Pipeline Architecture

```
  SRS Document  (.txt / .pdf / .docx)
        |
        |  "sample_srs.txt"  — Healthcare EHR Vital Signs Module SRS
        v
+-----------------------------------------------------------------------+
|  INGESTION   pipeline/ingestion.py                                    |
|  Model: spaCy en_core_web_sm (~12MB)                                  |
|                                                                       |
|  pdfplumber / python-docx / open()  ->  raw text                     |
|  spaCy sentence boundary detection  ->  sentence list                |
|  Strip section numbering (regex)    ->  clean sentences              |
|  Discard segments < 10 chars                                         |
|                                                                       |
|  ACTUAL RESULT: 32 clean sentences loaded                             |
+-----------------------------+-----------------------------------------+
                              |  32 clean sentences
                              v
+-----------------------------------------------------------------------+
|  GEMINI SRS ANALYZER   pipeline/gemini_srs_analyzer.py               |
|  Model: gemini-2.5-flash  (Google Generative AI API)                 |
|                                                                       |
|  Single API call with full SRS text in the prompt.                   |
|  Returns structured JSON with two top-level keys:                    |
|                                                                       |
|  "knowledge_base": {                                                  |
|      "params": [...],   "roles": [...],                              |
|      "workflows": [...], "time_constraints": [...]                   |
|  }                                                                    |
|  "sentences": [                                                       |
|      {                                                                |
|        "id": 1, "text": "...",                                        |
|        "label": "TESTABLE"|"DOMAIN_KNOWLEDGE"|"NOT_TESTABLE",        |
|        "confidence": 0.0-1.0,                                        |
|        "entities": { ACTION, USER_ROLE, CLINICAL_PARAM, VALUE,       |
|                      OPERATOR, CONDITION, TIME_CONSTRAINT },          |
|        "expected_result": "..."                                       |
|      }, ...                                                           |
|  ]                                                                    |
|  Parser normalises entity lists and values to snake_case.            |
+--------------+--------------------------------------------------------+
               |  GeminiAnalysisResult
               |
     +---------+---------+
     |                   |
     v                   v
+----------+     +---------------------------------------------------+
| STAGE 0  |     |  STAGE 1A — Testability Filter                    |
| Bootstrap|     |  pipeline/stage1/testability_filter.py            |
|          |     |                                                   |
| Reads    |     |  Reads analyzer.result.sentences[].label         |
| analyzer |     |  -> ClassifiedSentence(text, label, confidence)  |
| .result  |     +------------------------+--------------------------+
| .knowled-|                              |  32 ClassifiedSentence objects
| ge_base  |                              v
| ->       |     +---------------------------------------------------+
| SessionKB|     |  STAGE 1B — NER Extractor                         |
|          |     |  pipeline/stage1/ner_extractor.py                |
|          |     |  Reads entities + expected_result per sentence   |
|          |     |  NERResult now carries test_strategies: list     |
+----------+     +------------------------+--------------------------+
                                          |  32 NERResult objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 1C — Rule Assembler                        |
                 |  pipeline/stage1/rule_assembler.py               |
                 |                                                   |
                 |  NERResult -> FormalRule:                         |
                 |    1. Resolve ACTION via SessionKB               |
                 |    2. Resolve USER_ROLE (authorized/unauthorized) |
                 |    3. Pair CLINICAL_PARAM + OPERATOR + VALUE     |
                 |    4. event_based params kept (not dropped)      |
                 |    5. _resolve_strategies(): Gemini strategies   |
                 |       -> enum values; heuristic fallback if empty|
                 |    6. FormalRule strategies: up to 6 types        |
                 +------------------------+--------------------------+
                                          |  32 FormalRule objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 2 — Scenario Generator                     |
                 |  pipeline/stage2/scenario_generator.py           |
                 |                                                   |
                 |  BVA               — boundary math                |
                 |  EP                — role equivalence classes     |
                 |  STATE_TRANSITION  — state machine traversal      |
                 |  DECISION_TABLE    — 2^N condition combinations   |
                 |  TEMPORAL          — deadline/timing compliance   |
                 |  CLINICAL_VALIDATION — out-of-range input fencing |
                 |                                                   |
                 |  Fully KB-driven — no hardcoded roles/keywords    |
                 +------------------------+--------------------------+
                                          |  ~67 Scenario objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 3A — TC Generator                          |
                 |  pipeline/stage3/tc_generator.py                 |
                 |                                                   |
                 |  Scenario + FormalRule -> ISO29119TestCase        |
                 |  Step templates for all 6 strategies             |
                 |  Time constraint prefix dedup (re.sub fix)       |
                 |  All ISO 29119-3 mandatory fields populated      |
                 +------------------------+--------------------------+
                                          |  67 ISO29119TestCase objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 3B — Augmentor  [OPTIONAL]                 |
                 |  pipeline/stage3/augmentor.py                    |
                 |  Requires: --augment flag + OPENAI_API_KEY       |
                 +------------------------+--------------------------+
                                          |
                                          v
                               output/test_cases.json   (67 TCs)
                               output/test_cases.txt    (human-readable)
```

---

## 4. Key Data Structures

### 4.1 GeminiAnalysisResult

```python
@dataclass
class GeminiAnalysisResult:
    knowledge_base: dict    # params, roles, workflows, time_constraints
    sentences:      list    # one dict per sentence; text/label/confidence/entities/expected_result
    raw_json:       str     # original API response text (for debugging)
    expected_result: str        # Gemini-generated verifiable outcome

@dataclass
class GeminiAnalysisResult:
    sentences: List[SentenceAnalysis]
    knowledge_base: dict        # params, roles, workflows, time_constraints
```

### 4.2 SessionKB

Central context object built by Stage 0 from `analyzer.result.knowledge_base`.

| Type               | Dataclass     | Key fields                                                         |
|--------------------|---------------|--------------------------------------------------------------------|
| Clinical parameter | ClinicalParam | name, unit, critical_low, critical_high, normal_low, normal_high   |
| User role          | UserRole      | name, authorized (bool), capabilities, access_level               |
| Workflow           | WorkflowDef   | states, valid_transitions, terminal_states, trigger_keywords       |
| Time constraint    | dict          | value, unit, source sentence                                       |

**`state_transition_indicator_words` property** is now fully dynamic:

```python
@property
def state_transition_indicator_words(self) -> list[str]:
    if self._workflows:
        words = set()
        for wf in self._workflows.values():
            words.update(s.lower() for s in wf.states)
            words.update(kw.lower() for kw in wf.trigger_keywords)
        return list(words)
    return self._state_transition_words   # fallback: registry list, pre-bootstrap only
```

No hardcoded transition words. Derived entirely from workflow data in the SRS.

### 4.3 NERResult

```python
@dataclass
class NERResult:
    sentence:        str
    entities:        dict   # 7 keys: ACTION, USER_ROLE, CLINICAL_PARAM, OPERATOR,
                            #         CLINICAL_VALUE, TIME_CONSTRAINT, CONDITION
    test_strategies: list   # Gemini-recommended strategy names (uppercase strings)
    expected_result: str    # Gemini-generated verifiable outcome
```

### 4.4 FormalRule

```python
@dataclass
class FormalRule:
    rule_id:          str
    source_sentence:  str
    action:           str
    conditions:       list   # [{param, operator, value, unit, abstract, source}]
    user_role:        object
    result:           str
    time_constraint:  object
    strategies:       list   # List[TestStrategy] — 1–3 strategies per rule
    has_abstract_terms: bool
```

### 4.5 TestStrategy Enum

```python
class TestStrategy(Enum):
    BVA                 = "BVA"
    EP                  = "EP"
    STATE_TRANSITION    = "STATE_TRANSITION"
    DECISION_TABLE      = "DECISION_TABLE"
    TEMPORAL            = "TEMPORAL"
    CLINICAL_VALIDATION = "CLINICAL_VALIDATION"
```

### 4.6 ISO29119TestCase

```python
@dataclass
class ISO29119TestCase:
    tc_id:               str
    purpose:             str
    priority:            str          # HIGH | MEDIUM | LOW
    classification:      str
    preconditions:       list
    dependencies:        list
    inputs:              dict
    steps:               list
    expected_result:     str
    suspension_criteria: str
    postconditions:      str
    source_requirement:  str
    rule_id:             str
    scenario_id:         str
    strategy:            str          # BVA | EP | STATE_TRANSITION | DECISION_TABLE |
                                      # TEMPORAL | CLINICAL_VALIDATION
```

---

## 5. Gemini Integration Details

### config.py

```python
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "<your-key>")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LINGUISTIC_REGISTRY_PATH = "linguistic_registry"
OUTPUT_DIR = "output"
BVA_WELL_INSIDE_OFFSET = 5
OPENAI_MODEL = "gpt-4o"
EDGE_CASES_PER_TC = 5
```

### Prompt schema sent to Gemini

The system instruction instructs Gemini to act as a senior QA engineer and return **only valid
JSON** with this structure per sentence:

```json
{
  "id": 1,
  "text": "...",
  "label": "TESTABLE",
  "confidence": 0.9,
  "entities": {
    "ACTION":           ["alert"],
    "USER_ROLE":        ["nurse"],
    "CLINICAL_PARAM":   ["spo2"],
    "CLINICAL_VALUE":   ["90"],
    "OPERATOR":         ["<"],
    "CONDITION":        ["when"],
    "TIME_CONSTRAINT":  ["within 5 seconds"]
  },
  "expected_result": "critical alert is displayed within 5 seconds"
}
```

**CLINICAL_PARAM definition includes event-based params** (not just numeric values):

- Numeric: `spo2`, `heart_rate`, `blood_pressure`, `respiratory_rate`, `glucose`
- Event-based: `drug_interaction`, `duplicate_order`, `session_inactivity`,
  `conflicting_medication`, `authentication_factor`

**OPERATOR for events:** Use `"="` for non-numeric triggers (e.g. `drug_interaction = detected`).

**CLINICAL_VALUE for events:** Use `"detected"`, `"true"`, `"simultaneous"`.

### Why Gemini Replaced the Local Model Stack

| Concern                  | Local models                          | Gemini                                     |
|--------------------------|---------------------------------------|--------------------------------------------|
| Testability accuracy     | 25% of sentences classified TESTABLE  | 100% correctly handled                     |
| Drug interaction         | 0 TCs — NLI missed it                 | TCs generated (event-based EP)             |
| BVA thresholds           | 0 BVA TCs — BioGPT below confidence   | BVA TCs from Gemini-extracted thresholds   |
| Download size            | ~2GB+ (biogpt + NER + NLI)            | 0 MB (API-only)                            |
| Speed                    | 110+ seconds                          | ~70 seconds                                |
| Cross-sentence context   | None                                  | Full document context in one call          |

---

## 6. Strategy Selection — `_resolve_strategies()`

**File:** `pipeline/stage1/rule_assembler.py`

Replaced the old `_detect_strategies()` heuristic (keyword matching) with
`_resolve_strategies()`, which uses a direct string-to-enum mapping:

```python
_STRATEGY_NAME_MAP: dict[str, TestStrategy] = {s.value.upper(): s for s in TestStrategy}

def _resolve_strategies(gemini_strats, conditions, time_constraint, skb):
    # Primary path: Gemini's string names -> TestStrategy enums directly
    if gemini_strats:
        resolved = [_STRATEGY_NAME_MAP[n.upper()] for n in gemini_strats
                    if n.upper() in _STRATEGY_NAME_MAP]
        if resolved:
            return resolved

    # Fallback ONLY when Gemini returns nothing (e.g. old cached response):
    if numeric_conditions:  # -> [BVA]
    elif time_constraint:   # -> [TEMPORAL]
    elif kb has workflows:  # -> [STATE_TRANSITION]
    else:                   # -> [EP]
```

Gemini is the authoritative source. The fallback is a minimal single-strategy safety net used
only when `gemini_strats` is empty (e.g. when `gemini_srs_analyzer.py` is reverted to a
version that does not emit the `test_strategies` field).

---

## 7. Bug Fixes Applied

### 7.1 BVA boundary_exact Wrong Expected Result

**Root cause:** At threshold T with operator `<`, condition `value < T` is **FALSE** when
`value == T`. The old code returned `boundary_exact:{result}` implying the alert triggered.

**Fix:**
```python
# operator "<": exact boundary → condition is FALSE → no trigger
scenarios.append(make_sc("boundary_exact", threshold, f"no_{rule.result}", "HIGH"))
```
Same fix applied to `>` operator.

### 7.2 EP TC Wrong Role (nurse instead of lab_technician)

**Root cause:** `valid_role` was validated against `authorized_roles` from KB. If the
SRS-named role (e.g. `lab_technician`) was absent from the KB authorized list, it fell back
to `authorized_roles[0]` (e.g. `nurse`).

**Fix:**
```python
# Always use the role the SRS explicitly names for this rule
valid_role = rule.user_role if rule.user_role else (
    authorized_roles[0] if authorized_roles else None
)
```

### 7.3 STATE_TRANSITION = 0 (workflow key mismatch)

**Root cause:** Old `_detect_workflow()` returned `"medication_order"` but SessionKB registered
the workflow as `"medication_order_workflow"` — key mismatch, `kb.get_workflow()` returned None.

**Fix:** Replaced with `kb.detect_workflow_from_text(rule.source_sentence)` which scores
against actual KB-registered state names and trigger_keywords.

### 7.4 Missing Requirements (blocked transitions)

**Root cause:** Gemini misclassified "prevent transitioning" sentences as DOMAIN_KNOWLEDGE.

**Fix:** Added explicit TESTABLE classification bullets:
- "shall prevent", "shall not", "cannot", "not be able to"
- workflow blocked transitions, prohibition of state jumps

---

## 8. Hardcoding Removal

All domain-specific hardcoding has been removed. The pipeline is now fully dynamic.

| Location                | What was hardcoded                     | Now                                               |
|-------------------------|----------------------------------------|---------------------------------------------------|
| `scenario_generator.py` | `_WORKFLOW_KEYWORDS` dict              | Deleted — uses `kb.detect_workflow_from_text()`   |
| `scenario_generator.py` | `_detect_workflow()` function          | Deleted                                           |
| `scenario_generator.py` | `boundary_roles = ["ward_manager"]`    | Deleted — uses `kb.get_all_roles()`               |
| `scenario_generator.py` | `rule.user_role or "physician"`        | Deleted — skips if no role                        |
| `scenario_generator.py` | `unauthorized[0] if ... else "patient"`| Deleted — skips if KB has nothing                 |
| `session_kb.py`         | `state_transition_indicator_words` list| Dynamic: derived from wf.states + trigger_keywords|
| `linguistic_registry`   | `state_transition_indicator_words`     | Removed — comment explains it is runtime-derived  |
| `tc_generator.py`       | `"clinician"` fallback in steps        | Replaced with `"an authorized user"`              |
| `rule_assembler.py`     | `_detect_strategies()` heuristic       | Replaced with `_resolve_strategies()` + enum map  |

**One intentional exception:** `_INSTRUMENT_MAX` dict in `scenario_generator.py` is kept — it
contains physiological instrument physical maximums (e.g. SpO2 max = 100%), not domain business
logic. These are universal physical constraints, not SRS-specific.

---

## 9. Test Strategy Details

### BVA (Boundary Value Analysis)

**Triggered when:** Rule has a numeric threshold + comparison operator.

For threshold T and operator `<`:

| Scenario            | Value  | Expected              | Priority |
|---------------------|--------|-----------------------|----------|
| boundary_below      | T−1    | trigger result        | HIGH     |
| boundary_exact      | T      | `no_{result}` (fixed) | HIGH     |
| boundary_above      | T+1    | `no_{result}`         | MEDIUM   |
| well_inside_valid   | T−5    | trigger result        | MEDIUM   |
| well_inside_invalid | T+5    | `no_{result}`         | LOW      |

Offset: `BVA_WELL_INSIDE_OFFSET = 5` in config.py.

### EP (Equivalence Partitioning)

**Triggered when:** Rule has a user role OR access-control language OR event-based condition.

Uses SessionKB role lists — no hardcoded role names:

```
valid_class    -> rule.user_role (SRS-named role, always primary)    -> MEDIUM
invalid_class  -> first unauthorized role from KB                     -> HIGH
boundary_class -> first remaining KB role not already used            -> MEDIUM
```

### STATE_TRANSITION

**Triggered when:** Sentence scores highest against a registered workflow in SessionKB.

Workflow detection uses `kb.detect_workflow_from_text()` — scores by matching state names and
trigger_keywords registered from the SRS. No hardcoded word list.

### DECISION_TABLE

**Triggered when:** Rule has 2+ conditions. Generates 2^N true/false combinations.

### TEMPORAL

**Triggered when:** Rule contains a time constraint (deadline, timeout, response-time SLA).

Generates two scenarios per rule:
- `compliant`: response within deadline → success
- `non_compliant`: response exceeds deadline → timeout/breach action

Step templates include clock-sync preconditions and timestamp recording steps.

### CLINICAL_VALIDATION

**Triggered when:** Rule requires input validation (null, out-of-range, invalid format).

Generates scenarios with instrument-maximum upper bounds (`_INSTRUMENT_MAX`) and null/empty
inputs to verify the system rejects invalid clinical data.

---

## 10. Event-Based Parameter Handling

Rules whose `CLINICAL_PARAM` is not in the SessionKB numeric params are kept as event-based
rather than dropped:

```python
if raw_param in kb_params:
    canonical_param = kb_params[raw_param]
    source = "srs"
else:
    canonical_param = raw_param      # keep as-is
    source = "event_based"
```

Fallback condition synthesis when list is empty but ACTION + CONDITION both exist:
```python
conditions.append({
    "param": f"{primary_action}_trigger",
    "operator": "=", "value": "detected", "source": "event_based"
})
```

---

## 11. TC Generator — Step Templates

**File:** `pipeline/stage3/tc_generator.py`

All 6 strategies have dedicated step templates in `_infer_steps()`:

| Strategy            | Steps pattern                                                              |
|---------------------|----------------------------------------------------------------------------|
| STATE_TRANSITION    | Login → navigate to workflow → locate record → attempt transition → observe |
| ALERT/NOTIFY        | Login → navigate dashboard → set parameter → wait → observe alert panel    |
| RESTRICT/PREVENT    | Login → attempt restricted action → observe denial → verify no data exposed |
| LOG/RECORD          | Login → perform action → navigate audit log → verify log entry             |
| TEMPORAL            | Configure monitoring → login → trigger → record timestamps → compare deadline |
| CLINICAL_VALIDATION | Login → navigate data entry → enter value → attempt submit → verify rejection |

Time constraint dedup fix:
```python
tc_str = re.sub(r'^(?:within|after|in)\s+', '', tc_str, flags=re.IGNORECASE)
expected_str += f" (within {tc_str})"
```

---

## 12. Output Files

`main.py` writes both files directly — no external utility script needed:

| File                      | Format          | Contents                              |
|---------------------------|-----------------|---------------------------------------|
| `output/test_cases.json`  | JSON array      | Full ISO 29119-3 structure, all TCs   |
| `output/test_cases.txt`   | Human-readable  | Formatted TC blocks for QA review     |

---

## 13. Benchmarks and Evaluation Datasets

### Public SRS Datasets (usable as pipeline input)

| Dataset                        | Domain              | Size    | Source                  |
|-------------------------------|---------------------|---------|-------------------------|
| PROMISE SRS corpus             | Mixed / healthcare  | ~15 docs| openscience.us/repo     |
| PURE dataset                   | Mixed               | 79 docs | github.com/AlDanial/PURE|
| ONC EHR Certification Criteria | Healthcare EHR      | Large   | healthit.gov            |
| HL7 FHIR specifications        | Healthcare          | Large   | fhir.org                |
| IEEE 830 sample SRS docs       | Generic             | Several | IEEE Std 830 appendices |
| PROMISE CM1                    | Medical device/NASA | 505 reqs| Defect-labeled          |

### Evaluation Benchmarks

| Benchmark  | Measures                                             | Relevance |
|------------|------------------------------------------------------|-----------|
| LLM4Fin    | TC generation for financial SRS (coverage, diversity)| High      |
| TCRAFT     | TC quality from NL requirements (coverage + mutation)| Medium    |
| TCGen-Eval | LLM TC quality (BLEU/ROUGE + structural validity)    | Medium    |

### Recommended Evaluation Metrics

- **Coverage**: % of TESTABLE sentences producing ≥1 TC (current: 100%)
- **Strategy diversity**: distribution across 6 strategies (tracked in output)
- **Structural validity**: all ISO 29119-3 mandatory fields present (enforced by dataclass)
- **Manual expert review**: QA engineer scores N random TCs (most credible for a paper)
- **Mutation testing**: inject known bugs; check TC detection rate

---

## 14. Configuration

| Setting                  | Value               | Purpose                                   |
|--------------------------|---------------------|-------------------------------------------|
| GEMINI_API_KEY           | env or config.py    | Google Gemini API key                     |
| GEMINI_MODEL             | gemini-2.5-flash    | Gemini model name                         |
| LINGUISTIC_REGISTRY_PATH | linguistic_registry | Path to registry JSON                     |
| OUTPUT_DIR               | output              | Output directory for JSON/TXT             |
| BVA_WELL_INSIDE_OFFSET   | 5                   | Distance from boundary for BVA values     |
| OPENAI_MODEL             | gpt-4o              | GPT model for augmentation (optional)     |
| EDGE_CASES_PER_TC        | 5                   | Edge cases per TC from GPT-4o             |

---

## 15. How to Run

```powershell
# Set API key (do not commit to git)
$env:GEMINI_API_KEY = "your-key-from-aistudio.google.com"

# Basic run
python main.py --input sample_srs.txt

# Verbose output
python main.py --input sample_srs.txt --verbose

# With GPT-4o edge case augmentation
$env:OPENAI_API_KEY = "sk-..."
python main.py --input sample_srs.txt --augment
```

Output files written automatically to `output/test_cases.json` and `output/test_cases.txt`.

---

## 16. Dependencies

| Package        | Version  | Purpose                                                    |
|----------------|----------|------------------------------------------------------------|
| google-genai   | >=0.7.0  | Gemini API — classification, NER, knowledge extraction     |
| spacy          | >=3.7.0  | Sentence segmentation (en_core_web_sm)                     |
| pdfplumber     | >=0.10.0 | PDF text extraction                                        |
| python-docx    | >=1.1.0  | DOCX text extraction                                       |
| openai         | >=1.0.0  | GPT-4o augmentation (optional, --augment only)             |
| tqdm           | >=4.66.0 | Progress bars                                              |

**No PyTorch, no transformers, no BioGPT, no HuggingFace NER models required.**

---

## 17. Known Issues / Pending Items

| Issue                              | Status          | Notes                                                    |
|------------------------------------|-----------------|----------------------------------------------------------|
| API quota exhaustion (429)         | Active          | Free-tier daily quota; wait or upgrade to pay-as-you-go  |
| Gemini response caching            | Not implemented | Hash SRS → cache JSON → skip API on repeat runs          |
| `gemini_srs_analyzer.py` reverted  | Done by user    | `test_strategies` field removed from Gemini output;      |
|                                    |                 | rule_assembler uses heuristic fallback for strategies    |
| LLM4Fin benchmark comparison       | Not run         | Need financial SRS dataset + evaluation harness          |
| Augmentor (GPT-4o edge cases)      | Optional        | Not tested in recent runs                                |
