# pipeline/stage1/testability_filter.py
#
# Testability classification - Gemini-powered.
# NO NLI model. NO regex override. NO linguistic_registry lookup.

from dataclasses import dataclass
from pipeline.gemini_srs_analyzer import GeminiSRSAnalyzer


@dataclass
class ClassifiedSentence:
    text: str
    label: str        # "TESTABLE", "DOMAIN_KNOWLEDGE", "NOT_TESTABLE"
    confidence: float
    reasoning: str = ""   # Gemini one-sentence explanation


def classify_sentences(
    sentences: list[str],
    analyzer: GeminiSRSAnalyzer,
) -> list[ClassifiedSentence]:
    """
    Return per-sentence testability classifications from the Gemini analysis.

    Args:
        sentences : All SRS sentence strings.
        analyzer  : GeminiSRSAnalyzer with .analyze() already called.

    Returns:
        List of ClassifiedSentence objects, one per input sentence, in order.
    """
    result = analyzer.result   # already computed - no new API call
    if result is None:
        return [
            ClassifiedSentence(text=s, label="DOMAIN_KNOWLEDGE", confidence=0.5)
            for s in sentences
        ]

    sent_map: dict[str, dict] = {s["text"]: s for s in result.sentences}

    classified: list[ClassifiedSentence] = []
    for sentence in sentences:
        data = sent_map.get(sentence)
        if data is None:
            classified.append(ClassifiedSentence(
                text=sentence,
                label="DOMAIN_KNOWLEDGE",
                confidence=0.5,
                reasoning="Not returned by Gemini - defaulted to DOMAIN_KNOWLEDGE.",
            ))
        else:
            classified.append(ClassifiedSentence(
                text=sentence,
                label=data.get("classification", "DOMAIN_KNOWLEDGE"),
                confidence=0.95,
                reasoning=data.get("reasoning", ""),
            ))

    return classified
