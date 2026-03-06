# TestGenX v3.0 -- Full Project Analysis (Updated)

---

## 1. What the Project Does

TestGenX is an **automated test case generator for healthcare software**. Given a plain-text,
PDF, or DOCX Software Requirements Specification (SRS) document, it reads the requirements and
outputs a structured JSON file of test cases that fully conform to **ISO/IEC/IEEE 29119-3** --
the international standard for software test documentation.

**The goal:** A QA engineer drops in an SRS file and gets professional, prioritised, traceable
test cases out -- zero manual writing, zero hardcoded knowledge.

### Project Overview

TestGenX reads a healthcare SRS document, sends the full text to Google Gemini in a single API
call, and gets back two things: a structured knowledge base (clinical parameters with numeric
thresholds, user roles with access levels, workflow state machines, and time constraints) and a
per-sentence analysis (testability label, named entity extraction across 7 entity types, and a
Gemini-written expected result string). That knowledge base is loaded into a central SessionKB
object, and the per-sentence entity data is assembled into FormalRule objects -- one per
testable requirement -- each tagged with one or more test strategies (BVA for numeric
thresholds, EP for role-based access control, STATE_TRANSITION for workflow state machines,
DECISION_TABLE for multi-condition logic). A scenario generator then expands each rule into
concrete test scenarios using the strategy algorithms (e.g. boundary math for BVA, role
equivalence classes for EP, state graph traversal for STATE_TRANSITION), and a final template
engine turns every scenario into a fully-populated ISO 29119-3 test case with purpose,
preconditions, numbered steps, expected result, suspension criteria, and full traceability back
to the source SRS sentence. The end result is a JSON file and a human-readable text file, both
containing professional test cases that a QA team can execute directly -- generated in under 90
seconds from a plain text requirements document.

### Verified run result (current -- Gemini pipeline)

```
Input   : sample_srs.txt
Runtime : ~70 seconds
Outputs : output/test_cases.json
          output/test_cases.txt

Sentences processed : 30
Testable sentences  : 30  (100% -- Gemini classifies all sentences)
Rules assembled     : 30  -> active rules produce scenarios
Scenarios generated : 57
Test cases output   : 57

Strategies  : BVA=10   EP=35   STATE_TRANSITION=12
Priorities  : HIGH=24  MEDIUM=31  LOW=2
```

### Before / After comparison

| Metric                     | Before (NLI + BioGPT + NER-ML)       | After (Gemini)                  |
|----------------------------|--------------------------------------|----------------------------------|
| Testable sentences         | 8 / 32 (25%)                         | 30 / 30 (100%)                  |
| Active rules               | 5                                    | 30                              |
| Test cases                 | 16                                   | **57**                          |
| BVA test cases             | 0                                    | **10**                          |
| STATE_TRANSITION TCs       | 4                                    | **12**                          |
| Drug interaction TCs       | 0                                    | **yes (sentence 18)**           |
| Models required            | transformers + torch + biogpt + ML   | **google-generativeai only**    |
| Time-constraint formatting | "within within 5 seconds" bug        | **Clean: (within 5 seconds)**   |

---

## 2. Project Structure

```
files/
+-- main.py                         Entry point -- orchestrates the full pipeline
+-- config.py                       All tunable settings + API key
+-- log_tcs.py                      Utility: print + save test cases from JSON  * NEW
+-- linguistic_registry             JSON -- NLP operator/action/condition patterns
+-- sample_srs.txt                  Example SRS input (Healthcare EHR SRS)
+-- requirements.txt                Python dependencies
+-- README.md                       Project readme
+-- project_analysis.md             This file
+-- output/
|   +-- test_cases.json             Generated test cases (ISO 29119-3)
|   +-- test_cases.txt              Human-readable formatted test cases  * NEW
+-- pipeline/
    +-- ingestion.py                Document reader + sentence splitter (spaCy)
    +-- session_kb.py               Central context object (SessionKB)
    +-- gemini_srs_analyzer.py      Gemini API wrapper -- classification + NER  * REWRITTEN
    |
    +-- stage0/                     SRS Bootstrap -- domain knowledge extraction
    |   +-- __init__.py
    |   +-- srs_bootstrap.py        Reads Gemini knowledge_base -> populates SessionKB
    |   +-- clinical_retriever.py   (Legacy -- no longer called)
    |
    +-- stage1/                     Rule Extraction
    |   +-- __init__.py
    |   +-- testability_filter.py   Reads Gemini classification results  * UPDATED
    |   +-- ner_extractor.py        Reads Gemini NER results + expected_result  * UPDATED
    |   +-- rule_assembler.py       NERResult -> FormalRule; event-based params  * UPDATED
    |
    +-- stage2/                     Scenario Generation
    |   +-- __init__.py
    |   +-- scenario_generator.py   BVA / EP / STATE_TRANSITION / DECISION_TABLE
    |   +-- knowledge_base.py       Legacy KB class (type reference only)
    |
    +-- stage3/                     Test Case Generation
        +-- __init__.py
        +-- tc_generator.py         Scenario -> ISO29119TestCase; time-fix  * UPDATED
        +-- augmentor.py            Optional GPT-4o edge case augmentation
```

---

## 3. Full Pipeline Architecture

```
  SRS Document  (.txt / .pdf / .docx)
        |
        |  "sample_srs.txt"  -- Healthcare EHR Vital Signs Module SRS
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
|  ACTUAL RESULT: 30 clean sentences loaded                             |
+-----------------------------+-----------------------------------------+
                              |  30 clean sentences
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
|        "id": 1,                                                       |
|        "text": "...",                                                 |
|        "label": "TESTABLE" | "DOMAIN_KNOWLEDGE" | "NOT_TESTABLE",   |
|        "confidence": 0.0-1.0,                                        |
|        "entities": {                                                  |
|            "ACTION": [...], "USER_ROLE": [...],                      |
|            "CLINICAL_PARAM": [...], "CLINICAL_VALUE": [...],         |
|            "OPERATOR": [...], "CONDITION": [...],                    |
|            "TIME_CONSTRAINT": [...]                                   |
|        },                                                             |
|        "expected_result": "..."   <- verifiable outcome string       |
|      }, ...                                                           |
|  ]                                                                    |
|                                                                       |
|  Parser normalises entity lists, pads missing sentences,             |
|  and normalises all values to snake_case.                            |
|                                                                       |
|  ACTUAL RESULT:                                                       |
|    30 sentences classified (all TESTABLE or DOMAIN_KNOWLEDGE)        |
|    knowledge_base populated with params, roles, workflows, times     |
+--------------+--------------------------------------------------------+
               |  GeminiAnalysisResult
               |
     +---------+---------+
     |                   |
     v                   v
+----------+     +---------------------------------------------------+
| STAGE 0  |     |  STAGE 1A -- Testability Filter                   |
| Bootstrap|     |  pipeline/stage1/testability_filter.py            |
|          |     |                                                   |
| Reads    |     |  Reads analyzer.result.sentences[].label         |
| analyzer |     |  -> ClassifiedSentence(text, label, confidence)  |
| .result  |     |                                                   |
| .knowled-|     |  ACTUAL RESULT: All 30 sentences classified      |
| ge_base  |     |  (Gemini decides TESTABLE vs DOMAIN_KNOWLEDGE)   |
| ->       |     +------------------------+--------------------------+
| SessionKB|                              |  30 ClassifiedSentence objects
|          |                              v
| ACTUAL:  |     +---------------------------------------------------+
| 5 params |     |  STAGE 1B -- NER Extractor                        |
| 4 roles  |     |  pipeline/stage1/ner_extractor.py                |
| 1 wflow  |     |                                                   |
| 3 times  |     |  Reads analyzer.result.sentences[].entities      |
+----------+     |  and .expected_result for each TESTABLE sentence |
                 |  -> NERResult(sentence, entities, expected_result)|
                 |                                                   |
                 |  ACTUAL RESULT: 30 NERResult objects             |
                 +------------------------+--------------------------+
                                          |  30 NERResult objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 1C -- Rule Assembler                       |
                 |  pipeline/stage1/rule_assembler.py               |
                 |                                                   |
                 |  NERResult -> FormalRule:                         |
                 |    1. Resolve ACTION via SessionKB               |
                 |    2. Resolve USER_ROLE (authorized/unauthorized) |
                 |    3. Pair CLINICAL_PARAM + OPERATOR + VALUE     |
                 |       -> If param not in KB: keep as event_based |
                 |    4. has_action_trigger fallback for events     |
                 |    5. expected_result from Gemini or _infer_result|
                 |    6. Tag strategies: BVA / EP / STATE_TRANSITION|
                 |                                                   |
                 |  ACTUAL RESULT: 30 FormalRule objects assembled  |
                 +------------------------+--------------------------+
                                          |  30 FormalRule objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 2 -- Scenario Generator                    |
                 |  pipeline/stage2/scenario_generator.py           |
                 |                                                   |
                 |  BVA  -- boundary math on numeric thresholds     |
                 |  EP   -- role-based equivalence classes          |
                 |  STATE_TRANSITION -- state machine traversal     |
                 |  DECISION_TABLE  -- condition combinations        |
                 |                                                   |
                 |  ACTUAL RESULT: 57 scenarios generated           |
                 |    BVA=10  EP=35  STATE_TRANSITION=12            |
                 +------------------------+--------------------------+
                                          |  57 Scenario objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 3A -- TC Generator                         |
                 |  pipeline/stage3/tc_generator.py                 |
                 |                                                   |
                 |  Scenario + FormalRule -> ISO29119TestCase        |
                 |  Time constraint prefix dedup (re.sub fix)       |
                 |  All ISO 29119-3 mandatory fields populated      |
                 |                                                   |
                 |  ACTUAL RESULT: 57 test cases                    |
                 |    TC-HEALTH-001 through TC-HEALTH-057           |
                 |    HIGH=24  MEDIUM=31  LOW=2                     |
                 +------------------------+--------------------------+
                                          |  57 ISO29119TestCase objects
                                          v
                 +---------------------------------------------------+
                 |  STAGE 3B -- Augmentor  [OPTIONAL]                |
                 |  pipeline/stage3/augmentor.py                    |
                 |  Requires: --augment flag + OPENAI_API_KEY       |
                 |  NOT RUN in this execution                        |
                 +------------------------+--------------------------+
                                          |
                                          v
                               output/test_cases.json   (57 TCs)
                               output/test_cases.txt    (human-readable)
```

---

## 4. Key Data Structures

### 4.1 GeminiAnalysisResult

```python
@dataclass
class SentenceAnalysis:
    id: int
    text: str
    label: str                  # "TESTABLE" | "DOMAIN_KNOWLEDGE" | "NOT_TESTABLE"
    confidence: float
    entities: dict              # 7 keys: ACTION, USER_ROLE, CLINICAL_PARAM,
                                #         CLINICAL_VALUE, OPERATOR, CONDITION, TIME_CONSTRAINT
    expected_result: str        # Gemini-generated verifiable outcome

@dataclass
class GeminiAnalysisResult:
    sentences: List[SentenceAnalysis]
    knowledge_base: dict        # params, roles, workflows, time_constraints
```

### 4.2 SessionKB

Central context object built by Stage 0 from `analyzer.result.knowledge_base`.

| Type               | Dataclass     | Key fields                                                    |
|--------------------|---------------|---------------------------------------------------------------|
| Clinical parameter | ClinicalParam | name, unit, critical_low, critical_high, normal_low, normal_high |
| User role          | UserRole      | name, authorized (bool), capabilities, access_level          |
| Workflow           | WorkflowDef   | states, valid_transitions, terminal_states                   |
| Time constraint    | dict          | value, unit, source sentence                                  |

### 4.3 NERResult

```python
@dataclass
class NERResult:
    sentence: str
    entities: dict          # same 7-key schema
    expected_result: str    # from Gemini  * NEW
```

### 4.4 FormalRule

```python
@dataclass
class FormalRule:
    rule_id: str
    action: str
    user_roles: List[str]
    conditions: List[dict]      # [{param, operator, value, source}]
    expected_result: str        # Gemini result or _infer_result() fallback
    strategies: List[str]       # ["BVA", "EP", "STATE_TRANSITION", ...]
    source_sentence: str
    time_constraint: Optional[str]
```

### 4.5 ISO29119TestCase

Full ISO/IEC/IEEE 29119-3 compliant record:

```python
@dataclass
class ISO29119TestCase:
    tc_id: str
    purpose: str
    priority: str               # HIGH | MEDIUM | LOW
    classification: str
    preconditions: List[str]
    dependencies: List[str]
    inputs: dict
    steps: List[str]
    expected_result: str
    suspension_criteria: str
    postconditions: str
    source_requirement: str
    rule_id: str
    scenario_id: str
    strategy: str               # BVA | EP | STATE_TRANSITION | DECISION_TABLE
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

### Why Gemini replaced the local model stack

| Concern                          | Local models                        | Gemini                                    |
|----------------------------------|-------------------------------------|-------------------------------------------|
| Testability accuracy             | 25% of sentences classified TESTABLE| All 30 sentences correctly handled        |
| Drug interaction (sentence 18)   | 0 TCs -- NLI missed it              | TCs generated                             |
| BVA thresholds                   | 0 BVA TCs -- BioGPT below confidence| 10 BVA TCs from Gemini-extracted thresholds|
| Download size                    | ~2GB+ (biogpt + NER + NLI)          | 0 MB (API-only)                           |
| Setup                            | transformers, torch, sacremoses     | google-generativeai                       |
| Speed                            | 110+ seconds                        | ~70 seconds                               |

---

## 6. Rule Assembler -- Event-Based Param Handling

The Rule Assembler (`pipeline/stage1/rule_assembler.py`) was updated to handle sentences whose
CLINICAL_PARAM does not appear in the SessionKB (e.g. `drug_interaction`, `session_inactivity`).

### Key changes

**Non-KB param handling (was: drop; now: keep as event_based)**

```python
if raw_param in kb_params:
    canonical_param = kb_params[raw_param]
    source = "srs"
else:
    canonical_param = raw_param          # keep as-is
    source = "event_based"               # mark source
```

**Default operator and value for events**

```python
operator = condition.get("operator") or "="
value    = condition.get("value")    or "detected"
```

**has_action_trigger keep criterion**

A rule is kept even if conditions list is empty, as long as ACTION + CONDITION are both present:

```python
has_action_trigger = bool(ner.entities.get("ACTION")) and bool(ner.entities.get("CONDITION"))
```

**Event condition synthesis fallback**

If conditions is still empty but ACTION + CONDITION exist:

```python
conditions.append({
    "param":    f"{primary_action}_trigger",
    "operator": "=",
    "value":    "detected",
    "source":   "event_based"
})
```

### Result for sentence 18 ("drug interaction warning when two conflicting medications")

```
CLINICAL_PARAM : ["drug_interaction"]    (event-based, not in KB)
ACTION         : ["warn"]
CONDITION      : ["when"]
expected_result: "drug interaction warning is displayed"

-> FormalRule assembled with:
    conditions = [{param: "drug_interaction", operator: "=", value: "detected"}]
    strategies = ["EP"]
    -> EP test cases generated
```

---

## 7. TC Generator -- Time Constraint Fix

**File:** `pipeline/stage3/tc_generator.py`

**Bug:** Time constraints from Gemini already contain `"within"` / `"after"` / `"in"` as
prefixes. The generator was prepending `"within"` again, producing:

- `"within within 5 seconds"`
- `"within after 10 minutes"`

**Fix:**

```python
import re

tc_str = re.sub(r'^(?:within|after|in)\s+', '', tc_str, flags=re.IGNORECASE)
expected_str += f" (within {tc_str})"
```

**Result:** Clean output `"critical alert displayed (within 5 seconds)"`.

---

## 8. Sample Test Cases

### TC-001 -- BVA (SpO2 threshold)

```
TC-HEALTH-001
Purpose      : Verify system BVA behavior: spo2 < 90 -- boundary below (value=89)
Priority     : HIGH
Strategy     : BVA
Preconditions: Patient monitoring active. SpO2 sensor connected.
Inputs       : {"spo2": 89}
Steps        :
  1. Set spo2 to 89 in the simulated patient monitoring environment.
  2. Trigger the monitoring check.
  3. Observe the system response.
Expected     : alert triggered -- spo2 = 89 is at boundary_below (within 5 seconds)
Source req   : The system shall alert the nurse when SpO2 drops below 90%.
```

### TC-033 -- EP (drug_interaction event-based)

```
TC-HEALTH-033
Purpose      : Verify system EP behavior for drug_interaction = detected (valid_class)
Priority     : MEDIUM
Strategy     : EP
Preconditions: EHR system running. Patient record exists with active medication list.
Inputs       : {"drug_interaction": "detected", "user_role": "physician"}
Steps        :
  1. Add two conflicting medications to a patient's medication list.
  2. Trigger the medication order.
  3. Observe the system response.
Expected     : drug interaction warning is displayed
Source req   : The system shall display a drug interaction warning when two conflicting
               medications are prescribed simultaneously.
```

### TC-048 -- STATE_TRANSITION (medication_order workflow)

```
TC-HEALTH-048
Purpose      : Verify STATE_TRANSITION: medication_order DRAFT -> SUBMITTED (valid)
Priority     : MEDIUM
Strategy     : STATE_TRANSITION
Preconditions: medication_order workflow active. Order in DRAFT state.
Inputs       : {"workflow": "medication_order", "from_state": "DRAFT", "to_state": "SUBMITTED"}
Steps        :
  1. Create a medication order in DRAFT state.
  2. Verify the order by a pharmacist.
  3. Submit the order.
  4. Observe the resulting state.
Expected     : Order transitions to SUBMITTED state successfully.
Source req   : A medication order shall transition from DRAFT to SUBMITTED only when verified.
```

---

## 9. Test Strategy Summary

### BVA (Boundary Value Analysis)

**Triggered when:** Rule has a numeric threshold + comparison operator.

For threshold T and operator `<`:

| Scenario           | Value | Expected       | Priority |
|--------------------|-------|----------------|----------|
| boundary_below     | T-1   | trigger        | HIGH     |
| boundary_exact     | T     | boundary_exact | HIGH     |
| boundary_above     | T+1   | no trigger     | MEDIUM   |
| well_inside_valid  | T-5   | trigger        | MEDIUM   |
| well_inside_invalid| T+5   | no trigger     | LOW      |

Offset controlled by `BVA_WELL_INSIDE_OFFSET = 5` in config.py.

### EP (Equivalence Partitioning)

**Triggered when:** Rule has a user role OR access-control language OR event-based condition.

Uses SessionKB role lists:

```
valid_class    -> physician (authorized=True)        -> MEDIUM
invalid_class  -> lab_technician (authorized=False)  -> HIGH
boundary_class -> ward_manager (boundary)            -> MEDIUM
```

### STATE_TRANSITION

**Triggered when:** Sentence contains state-transition indicator words.

SessionKB `medication_order` workflow:

```
Valid:   DRAFT -> SUBMITTED -> DISPENSED
Invalid: DRAFT -> DISPENSED (skip), DISPENSED -> DRAFT (from terminal)
CANCELLED is terminal -- cannot be re-approved
```

### DECISION_TABLE

**Triggered when:** Rule has 2+ conditions.
Generates 2^N combinations of true/false for all conditions.

---

## 10. Configuration -- config.py

| Setting                   | Value              | Purpose                                   |
|---------------------------|--------------------|-------------------------------------------|
| GEMINI_API_KEY            | env or hardcoded   | Google Gemini API key                     |
| GEMINI_MODEL              | gemini-2.5-flash   | Gemini model name                         |
| LINGUISTIC_REGISTRY_PATH  | linguistic_registry| Path to registry JSON                     |
| OUTPUT_DIR                | output             | Output directory for JSON/TXT             |
| BVA_WELL_INSIDE_OFFSET    | 5                  | Distance from boundary for BVA values     |
| OPENAI_MODEL              | gpt-4o             | GPT model for augmentation (optional)     |
| EDGE_CASES_PER_TC         | 5                  | Edge cases per TC from GPT-4o             |

---

## 11. How to Run

```powershell
# Basic run
python main.py --input sample_srs.txt

# With verbose output
python main.py --input sample_srs.txt --verbose

# View and save test cases
python log_tcs.py
# -> prints all 57 TCs to console
# -> saves to output/test_cases.txt

# With GPT-4o augmentation
$env:OPENAI_API_KEY = "sk-..."
python main.py --input sample_srs.txt --augment
```

**Output files:**

- `output/test_cases.json` -- machine-readable, full ISO 29119-3 structure
- `output/test_cases.txt` -- human-readable formatted output (generated by log_tcs.py)

---

## 12. Dependencies

| Package              | Version   | Purpose                                             |
|----------------------|-----------|-----------------------------------------------------|
| google-generativeai  | >=0.7.0   | Gemini API -- classification, NER, knowledge extraction |
| spacy                | >=3.7.0   | Sentence segmentation (en_core_web_sm)              |
| pdfplumber           | >=0.10.0  | PDF text extraction                                 |
| python-docx          | >=1.1.0   | DOCX text extraction                                |
| openai               | >=1.0.0   | GPT-4o augmentation (optional)                      |
| tqdm                 | >=4.66.0  | Progress bars                                       |

**No PyTorch, no transformers, no BioGPT, no HuggingFace NER models required.**
