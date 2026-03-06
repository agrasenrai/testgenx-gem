# pipeline/stage1/ner_extractor.py
#
# NER extraction — Gemini-powered.
#
# GeminiSRSAnalyzer already extracted all named entities for every TESTABLE
# sentence during the single Stage 0 API call.  This module reads those
# pre-computed entities from the cached GeminiAnalysisResult and wraps them
# in NERResult objects so Stage 1C (Rule Assembler) is unchanged.
#
# Entities produced per sentence:
#   ACTION, USER_ROLE, CLINICAL_PARAM, OPERATOR (math symbols: < > <= >= = !=),
#   CLINICAL_VALUE, TIME_CONSTRAINT, CONDITION
#
# NO biomedical NER model.  NO regex fallback.  NO hardcoded keyword lists.
# Gemini understands the sentence semantically; it does not keyword-match.

from dataclasses import dataclass, field
from pipeline.gemini_srs_analyzer import GeminiSRSAnalyzer


@dataclass
class NERResult:
    sentence: str
    entities: dict = field(default_factory=dict)
    # {"ACTION": [...], "USER_ROLE": [...], "CLINICAL_PARAM": [...],
    #  "OPERATOR": [...], "CLINICAL_VALUE": [...],
    #  "TIME_CONSTRAINT": [...], "CONDITION": [...]}
    test_strategies: list = field(default_factory=list)  # Gemini-recommended strategies
    expected_result: str = ""   # Gemini-generated verifiable outcome description


def extract_entities(
    sentences: list[str],
    analyzer: GeminiSRSAnalyzer,
) -> list[NERResult]:
    """
    Return NERResult objects for the given TESTABLE sentences from the
    Gemini analysis cache.

    Args:
        sentences : List of TESTABLE sentence strings.
        analyzer  : GeminiSRSAnalyzer with .analyze() already called.

    Returns:
        List of NERResult objects, one per sentence, in order.
    """
    result   = analyzer.result   # already computed — no API call here
    if result is None:
        # Should not happen in normal flow; fail gracefully
        return [NERResult(sentence=s) for s in sentences]

    # Build a lookup by sentence text for fast O(1) access
    sent_map: dict[str, dict] = {s["text"]: s for s in result.sentences}

    ner_results: list[NERResult] = []
    for sentence in sentences:
        data = sent_map.get(sentence)
        if data is None:
            # Sentence not found in Gemini output — return empty entities
            ner_results.append(NERResult(sentence=sentence))
            continue

        raw_ents: dict = data.get("entities", {})

        # Guarantee all expected keys are present as lists
        entities: dict = {}
        for key in ("ACTION", "USER_ROLE", "CLINICAL_PARAM", "OPERATOR",
                    "CLINICAL_VALUE", "TIME_CONSTRAINT", "CONDITION"):
            val = raw_ents.get(key, [])
            entities[key] = val if isinstance(val, list) else [val]

        # test_strategies: Gemini's recommendation for which test types apply
        raw_strats = data.get("test_strategies", [])
        if not isinstance(raw_strats, list):
            raw_strats = []

        expected_result = data.get("expected_result", "") or ""
        ner_results.append(NERResult(
            sentence=sentence,
            entities=entities,
            test_strategies=[s.strip().upper() for s in raw_strats if isinstance(s, str)],
            expected_result=expected_result.strip(),
        ))

    return ner_results

