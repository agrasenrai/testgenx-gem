# pipeline/gemini_srs_analyzer.py
#
# Gemini-powered SRS Analyzer — replaces Stage 0 (regex + BioGPT),
# Stage 1A (NLI testability filter) and Stage 1B (biomedical NER + regex).
#
# ONE API call per SRS document.  Gemini reads ALL sentences at once so it
# understands cross-sentence context (e.g. a threshold defined in sentence 3
# is referenced by a condition in sentence 12).
#
# Produces a single `GeminiAnalysisResult` object that is stored on the
# SessionKB and consumed by srs_bootstrap, testability_filter, and
# ner_extractor — all without loading any local ML model.

import json
import os
import textwrap
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Output schema (mirrors SessionKB + NERResult structures exactly)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeminiAnalysisResult:
    """
    The parsed output of a single Gemini call over the full SRS.

    knowledge_base  — dict with keys: clinical_params, roles, workflows,
                      time_constraints.  Feeds Stage 0 / SessionKB.

    sentences       — list[dict], one per input sentence.  Each dict has:
                        text, classification, reasoning, entities.
                      Feeds Stage 1A (testability filter) and
                      Stage 1B (NER extractor).
    """
    knowledge_base: dict = field(default_factory=dict)
    sentences:      list = field(default_factory=list)
    raw_json:       str  = ""          # original API response text (for debugging)


# ─────────────────────────────────────────────────────────────────────────────
# The master prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = textwrap.dedent("""\
    You are a world-class requirements engineering expert, trained on
    ISO/IEC/IEEE 29119 (software testing) and IEEE 29148 (requirements
    engineering). You analyse Software Requirements Specification (SRS)
    documents for any domain and extract structured information needed to
    generate ISO-compliant test cases.

    You understand requirement conventions across all writing styles:
      - Modal-verb style   : "The system shall …", "The system must …"
      - Passive obligation : "Access is restricted to …", "Users are required to …"
      - Conditional rule   : "When X occurs, the system …"
      - Direct constraint  : "Maximum response time: 5 seconds"
      - Table / list style : "3.1  Alert timeout  |  within 5 s"
      - Informal style     : "Doctors can view all records. Nurses cannot prescribe."

    Your output must be a single, valid JSON object — nothing else.
    Follow the schema and rules described in the user message exactly.
""")

# The JSON schema is embedded directly in the prompt so Gemini knows
# precisely which fields to produce and what each means.

_PROMPT_TEMPLATE = textwrap.dedent("""\
    Analyse the SRS sentences listed below and return a single JSON object
    with this EXACT schema.  No prose.  No markdown fences.  Only JSON.

    ════════════════════════════════════════════════════════
    REQUIRED JSON SCHEMA
    ════════════════════════════════════════════════════════

    {{
      "knowledge_base": {{

        "clinical_params": [
          {{
            "name"                     : "<canonical lowercase name>",
            "unit"                     : "<unit string or empty string>",
            "critical_low"             : <number or null>,
            "critical_low_evidence"    : "<sentence that states this, or null>",
            "critical_high"            : <number or null>,
            "critical_high_evidence"   : "<sentence that states this, or null>",
            "normal_low"               : <number or null>,
            "normal_low_evidence"      : "<sentence or 'clinical knowledge'>",
            "normal_high"              : <number or null>,
            "normal_high_evidence"     : "<sentence or 'clinical knowledge'>"
          }}
        ],

        "roles": [
          {{
            "name"         : "<snake_case role name>",
            "authorized"   : <true | false>,
            "capabilities" : ["<verb phrase the role CAN do>"],
            "denied_capabilities" : ["<verb phrase the role CANNOT do>"],
            "access_level" : "<full | read-only | limited | medication-only | lab-only | none>"
          }}
        ],

        "workflows": [
          {{
            "name"              : "<snake_case workflow name>",
            "states"            : ["STATE_A", "STATE_B", ...],
            "valid_transitions" : [["FROM", "TO"], ...],
            "terminal_states"   : ["STATE_X", ...],
            "trigger_keywords"  : ["<word linking sentences to this workflow>"]
          }}
        ],

        "time_constraints": [
          {{
            "value"   : <number>,
            "unit"    : "<seconds | minutes | hours | days>",
            "context" : "<the SRS sentence this came from>"
          }}
        ]
      }},

      "sentences": [
        {{
          "text"           : "<exact sentence text>",
          "classification" : "<TESTABLE | DOMAIN_KNOWLEDGE | NOT_TESTABLE>",
          "reasoning"      : "<one sentence explaining why — required>",
          "entities": {{
            "ACTION"          : ["<canonical lowercase action verb>"],
            "USER_ROLE"       : ["<snake_case role name>"],
            "CLINICAL_PARAM"  : ["<canonical lowercase param name>"],
            "OPERATOR"        : ["<SYMBOL ONLY: use <  >  <=  >=  =  !=  never words>"],
            "CLINICAL_VALUE"  : ["<number + unit, e.g. '90 %' or '150 bpm'>"],
            "TIME_CONSTRAINT" : ["<e.g. 'within 5 seconds'>"],
            "CONDITION"       : ["<trigger word: when / if / after / before / ...>"]
          }},
          "expected_result"  : "<what the system must do / produce as a verifiable outcome, e.g. 'alert displayed on nurse dashboard', 'access denied with error message', 'audit log entry created'>"
        }}
      ]
    }}

    ════════════════════════════════════════════════════════
    CLASSIFICATION RULES
    ════════════════════════════════════════════════════════

    TESTABLE:
      The sentence describes a verifiable, concrete system behaviour —
      regardless of the writing style.  Apply this to:
        • Any obligation the system must fulfil that can be verified by a test
          ("shall alert", "must display", "is required to log", "should restrict")
        • Any constraint with a measurable threshold or condition
          ("SpO2 below 90%", "response within 5 seconds")
        • Any access-control rule ("only physicians may …", "nurses cannot prescribe")
        • Any workflow transition rule ("order transitions from DRAFT to SUBMITTED")
      Do NOT require the word "shall" — use semantic meaning, not keywords.

    DOMAIN_KNOWLEDGE:
      Background information needed to understand the domain but that does
      not itself describe a testable system behaviour.  Examples:
        • Definitions ("SpO2 is measured as …", "CPOE stands for …")
        • Context ("This document describes …")
        • General domain facts ("Normal SpO2 range is 95–100% for healthy adults")
        • Version headers, section titles

    NOT_TESTABLE:
      Vague, ambiguous, or aspirational — cannot be verified by a concrete
      test case even in principle.  Apply rarely; most SRS sentences are
      either TESTABLE or DOMAIN_KNOWLEDGE.

    ════════════════════════════════════════════════════════
    ENTITY EXTRACTION RULES
    ════════════════════════════════════════════════════════

    CLINICAL_PARAM — ANY observable/measurable system parameter or trigger event.
      It is NOT limited to vital signs.  Include:
        • Physiological measurements:  "spo2", "heart_rate", "glucose"
        • Detectable system events:    "drug_interaction", "duplicate_order",
                                       "session_inactivity", "conflicting_medication"
        • Security/auth conditions:    "failed_login_attempts", "authentication_factor"
        • Workflow state inputs:       "mandatory_fields_completed", "order_status"
      Use lowercase snake_case.  If the sentence says "two conflicting medications"
      the param is "drug_interaction".  If it says "same medication prescribed twice"
      the param is "duplicate_order".
      Do NOT leave CLINICAL_PARAM empty for a TESTABLE sentence — there is always
      at least one triggering parameter.

    OPERATOR — for event-based params without a numeric threshold, use "=":
      "drug_interaction = detected", "duplicate_order = true"
      ALWAYS output the math symbol, never the word:
      "below" | "drops below" | "less than" | "falls below" → <
      "above" | "exceeds"     | "greater than"              → >
      "at most" | "no more than" | "not exceeding"          → <=
      "at least" | "no less than" | "minimum"               → >=
      "equals"  | "equal to" | "is exactly"                 → =
      "not equal to" | "different from"                     → !=

    CLINICAL_VALUE — include the unit for numeric values: "90 %", "150 bpm".
      For event-based params use a descriptive token: "detected", "true", "simultaneous".

    USER_ROLE — use snake_case: "ward_manager" not "ward manager".

    ACTION — use single canonical verb: "alert" not "trigger alert",
      "restrict" not "deny access", "log" not "record".
      Canonical set: alert | restrict | log | escalate | lock | transition |
                     notify | prevent | generate | display | require | verify |
                     authenticate | validate | record | send | check | monitor

    EXPECTED_RESULT — one concise string describing the verifiable system outcome
    for TESTABLE sentences.  Be specific: include what is produced and where.
      Bad:  "system alerts"   Good: "visual alert shown on nurse workstation"
      Bad:  "access denied"   Good: "login attempt blocked with error code 403"
    For DOMAIN_KNOWLEDGE / NOT_TESTABLE sentences use an empty string "".

    Do NOT invent entities not present in the sentence.
    Omit entity types with no matches (empty list is fine).

    ════════════════════════════════════════════════════════
    KNOWLEDGE BASE RULES
    ════════════════════════════════════════════════════════

    clinical_params:
      • Include every measurable clinical/technical parameter mentioned.
      • For thresholds EXPLICITLY stated in the SRS, fill critical_low /
        critical_high with the exact value and cite the evidence sentence.
      • For normal ranges NOT stated in the SRS, fill normal_low / normal_high
        from your clinical / domain knowledge and set evidence to
        "clinical knowledge".
      • Set to null only if you genuinely do not know the value.

    roles:
      • Include every human actor mentioned (authorized or not).
      • authorized = true if the role is GRANTED the primary system privilege.
      • authorized = false if the sentence DENIES or RESTRICTS the role.
      • A role can appear multiple times in the SRS with different rules;
        merge into one entry with combined capabilities / denied_capabilities.

    workflows:
      • Detect state machines from ALLCAPS state names or transition language.
      • CRITICAL — valid_transitions must list ONLY transitions that the SRS
        EXPLICITLY states are allowed.  Do not infer adjacency.
      • If a sentence says "prevent transitioning from X to Y without Z status"
        or "only when Z is approved" — this means X→Y is BLOCKED.
        The valid chain is X→Z→Y.  Add ["X","Z"] and ["Z","Y"] to valid_transitions.
        Do NOT add ["X","Y"] — that is the invalid/blocked transition.
      • Example: "prevent a medication order from transitioning from SUBMITTED
        to DISPENSED without APPROVED status" means:
          valid_transitions: [["DRAFT","SUBMITTED"],["SUBMITTED","APPROVED"],["APPROVED","DISPENSED"]]
          NOT valid: ["SUBMITTED","DISPENSED"]
      • terminal_states = states that appear only as targets, never as sources,
        OR are explicitly described as final / prevented from transitioning out.

    time_constraints:
      • Extract every "within N unit" or "after N unit" or "in N unit" pattern.
      • Normalise unit to: seconds | minutes | hours | days.

    ════════════════════════════════════════════════════════
    OUTPUT RULES
    ════════════════════════════════════════════════════════

    • Return ONE JSON object.  No explanation text. No markdown. No code fences.
    • Every input sentence must appear exactly once in "sentences", in order.
    • "text" must be the EXACT sentence string provided — do not paraphrase.
    • If a sentence has no extractable entities, return empty lists, not null.

    ════════════════════════════════════════════════════════
    SRS SENTENCES TO ANALYSE
    ════════════════════════════════════════════════════════

{sentences_block}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Analyser class
# ─────────────────────────────────────────────────────────────────────────────

class GeminiSRSAnalyzer:
    """
    Wraps the Gemini API call.  Call `analyze(sentences)` once per run.
    The result is cached on the instance and re-used by srs_bootstrap,
    testability_filter, and ner_extractor — no re-calling the API.
    """

    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self._api_key   = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._result: GeminiAnalysisResult | None = None

        if not self._api_key:
            raise EnvironmentError(
                "[GeminiSRSAnalyzer] GEMINI_API_KEY not set. "
                "Export it: $env:GEMINI_API_KEY = 'your-key'"
            )

        from google import genai
        self._genai  = genai
        self._client = genai.Client(api_key=self._api_key)

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze(self, sentences: list[str]) -> GeminiAnalysisResult:
        """
        Analyse all SRS sentences in ONE Gemini call.
        Result is cached — subsequent calls return the cached object.

        Args:
            sentences: All clean sentences from the SRS document (from Ingestion).

        Returns:
            GeminiAnalysisResult with knowledge_base and per-sentence data.
        """
        if self._result is not None:
            return self._result

        print(f"  [Gemini] Sending {len(sentences)} sentences to {self.model_name}...")

        # Build numbered sentence block for the prompt
        block_lines = []
        for i, s in enumerate(sentences, start=1):
            block_lines.append(f"[{i:03d}] {s}")
        sentences_block = "\n".join(block_lines)

        prompt = _PROMPT_TEMPLATE.format(sentences_block=sentences_block)

        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    max_output_tokens=65536,
                ),
            )
            raw_text = response.text.strip()
        except Exception as e:
            raise RuntimeError(f"[GeminiSRSAnalyzer] API call failed: {e}") from e

        self._result = self._parse_response(raw_text, sentences)
        print(f"  [Gemini] Analysis complete.")
        _log_summary(self._result)
        return self._result

    @property
    def result(self) -> GeminiAnalysisResult | None:
        """Direct access to the cached result (None until analyze() is called)."""
        return self._result

    # ── Internal parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw_text: str, original_sentences: list[str]) -> GeminiAnalysisResult:
        """
        Parse Gemini's JSON output into a GeminiAnalysisResult.
        Applies defensive repairs for common model output quirks.
        """
        # Strip accidental markdown fences if present
        text = raw_text
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = "\n".join(text.splitlines()[:-1])
        text = text.strip()

        try:
            data: dict = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"[GeminiSRSAnalyzer] JSON parse error: {e}\nRaw:\n{raw_text[:500]}")
            # Return a minimal safe fallback so the pipeline doesn't crash
            return _fallback_result(original_sentences)

        kb   = data.get("knowledge_base", {})
        sents = data.get("sentences", [])

        # ── Defensive checks on sentences ─────────────────────────────────────
        # If Gemini returned fewer sentence entries than inputs, pad with
        # DOMAIN_KNOWLEDGE placeholders so array indices stay aligned.
        if len(sents) < len(original_sentences):
            logger.warning(
                f"[GeminiSRSAnalyzer] Got {len(sents)} sentence entries for "
                f"{len(original_sentences)} input sentences — padding missing ones."
            )
            existing_texts = {s.get("text", "").strip() for s in sents}
            for orig in original_sentences:
                if orig.strip() not in existing_texts:
                    sents.append({
                        "text": orig,
                        "classification": "DOMAIN_KNOWLEDGE",
                        "reasoning": "Not returned by model — classified as domain knowledge by default.",
                        "entities": {
                            "ACTION": [], "USER_ROLE": [], "CLINICAL_PARAM": [],
                            "OPERATOR": [], "CLINICAL_VALUE": [],
                            "TIME_CONSTRAINT": [], "CONDITION": [],
                        },
                    })

        # ── Normalise entities in each sentence ───────────────────────────────
        for sent in sents:
            ents = sent.setdefault("entities", {})
            for key in ("ACTION", "USER_ROLE", "CLINICAL_PARAM", "OPERATOR",
                        "CLINICAL_VALUE", "TIME_CONSTRAINT", "CONDITION"):
                val = ents.get(key)
                if val is None:
                    ents[key] = []
                elif isinstance(val, str):
                    ents[key] = [val]   # model occasionally returns a bare string

            # Normalise user roles to snake_case
            ents["USER_ROLE"] = [
                r.strip().lower().replace(" ", "_") for r in ents["USER_ROLE"]
            ]
            # Normalise clinical params to lowercase
            ents["CLINICAL_PARAM"] = [
                p.strip().lower().replace(" ", "_") for p in ents["CLINICAL_PARAM"]
            ]
            # Normalise actions to lowercase
            ents["ACTION"] = [a.strip().lower() for a in ents["ACTION"]]

            # Normalise expected_result — guaranteed to be a string
            er = sent.get("expected_result", "")
            sent["expected_result"] = er.strip() if isinstance(er, str) else ""

            # Ensure classification is one of the three valid values
            cls = sent.get("classification", "DOMAIN_KNOWLEDGE").upper()
            if cls not in ("TESTABLE", "DOMAIN_KNOWLEDGE", "NOT_TESTABLE"):
                cls = "DOMAIN_KNOWLEDGE"
            sent["classification"] = cls

        # ── Normalise knowledge_base ──────────────────────────────────────────
        # Ensure lists are always lists, handle null knowledge_base gracefully
        for key in ("clinical_params", "roles", "workflows", "time_constraints"):
            if not isinstance(kb.get(key), list):
                kb[key] = []

        # Normalise clinical param names to lowercase/snake_case in KB too
        for cp in kb.get("clinical_params", []):
            cp["name"] = cp.get("name", "").strip().lower().replace(" ", "_")

        # Normalise role names
        for role in kb.get("roles", []):
            role["name"] = role.get("name", "").strip().lower().replace(" ", "_")

        return GeminiAnalysisResult(
            knowledge_base=kb,
            sentences=sents,
            raw_json=raw_text,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_result(sentences: list[str]) -> GeminiAnalysisResult:
    """
    Minimal safe result used when JSON parsing fails completely.
    Everything is DOMAIN_KNOWLEDGE so no rules are assembled and the
    pipeline exits cleanly with 0 test cases rather than crashing.
    """
    sents = []
    for s in sentences:
        sents.append({
            "text": s,
            "classification": "DOMAIN_KNOWLEDGE",
            "reasoning": "Fallback: Gemini response could not be parsed.",
            "entities": {
                "ACTION": [], "USER_ROLE": [], "CLINICAL_PARAM": [],
                "OPERATOR": [], "CLINICAL_VALUE": [],
                "TIME_CONSTRAINT": [], "CONDITION": [],
            },
        })
    return GeminiAnalysisResult(
        knowledge_base={
            "clinical_params": [],
            "roles": [],
            "workflows": [],
            "time_constraints": [],
        },
        sentences=sents,
    )


def _log_summary(result: GeminiAnalysisResult) -> None:
    kb = result.knowledge_base
    n_testable = sum(
        1 for s in result.sentences if s.get("classification") == "TESTABLE"
    )
    print(f"    Params found        : {len(kb.get('clinical_params', []))}")
    print(f"    Roles found         : {len(kb.get('roles', []))}")
    print(f"    Workflows found     : {len(kb.get('workflows', []))}")
    print(f"    Time constraints    : {len(kb.get('time_constraints', []))}")
    print(f"    TESTABLE sentences  : {n_testable} / {len(result.sentences)}")
