# pipeline/stage0/clinical_retriever.py
#
# Uses BioGPT (microsoft/biogpt) — trained on 15M PubMed abstracts —
# to retrieve clinical reference ranges for parameters whose thresholds
# are NOT explicitly stated in the SRS document.
#
# Contract:
#   - Returns (value, confidence) where confidence >= CLINICAL_CONFIDENCE_MIN
#   - Returns (None, 0.0) if the model cannot confidently answer
#   - (None, 0.0) → param is unresolvable → rule will be SKIPPED upstream
#   - No fallback. No guessing. Either the model knows it or we skip.

import re
import logging
from config import CLINICAL_MODEL, CLINICAL_CONFIDENCE_MIN, CLINICAL_MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

_tokenizer = None
_model     = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import BioGptTokenizer, BioGptForCausalLM
        print(f"  [ClinicalRetriever] Loading {CLINICAL_MODEL} (first run downloads ~1.5GB)...")
        _tokenizer = BioGptTokenizer.from_pretrained(CLINICAL_MODEL)
        _model     = BioGptForCausalLM.from_pretrained(CLINICAL_MODEL)
        _model.eval()
        print("  [ClinicalRetriever] Model loaded.")


def _generate(prompt: str) -> str:
    """Run BioGPT generation and return the completed text beyond the prompt."""
    import torch
    _load_model()
    inputs = _tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=CLINICAL_MAX_NEW_TOKENS,
            do_sample=False,            # greedy — deterministic
            num_beams=3,                # beam search for quality
            early_stopping=True,
        )
    full_text  = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip()
    return completion


def _extract_first_number(text: str) -> float | None:
    """Extract first numeric value from generated text."""
    matches = re.findall(r'\b(\d+\.?\d*)\b', text)
    if matches:
        return float(matches[0])
    return None


def _compute_confidence(generated: str, expected_unit: str | None) -> float:
    """
    Heuristic confidence based on:
    - Whether a number was found
    - Whether the expected unit appears
    - Whether the generation is coherent (not repetitive/empty)
    """
    if not generated or len(generated.strip()) < 3:
        return 0.0

    has_number = bool(re.search(r'\b\d+\.?\d*\b', generated))
    if not has_number:
        return 0.0

    confidence = 0.75  # base: found a number

    if expected_unit:
        unit_lower = expected_unit.lower()
        if unit_lower in generated.lower():
            confidence += 0.15  # unit matches

    # Penalize if repetitive (sign of degenerate generation)
    words = generated.lower().split()
    if len(words) > 5 and len(set(words)) < len(words) * 0.5:
        confidence -= 0.30

    return min(confidence, 1.0)


# ── Public API ────────────────────────────────────────────────────────────────

def get_critical_low(param_name: str, unit: str | None = None) -> tuple[float | None, float]:
    """
    Ask BioGPT: what is the critical low threshold for this clinical parameter?

    Returns:
        (value, confidence) — or (None, 0.0) if unresolvable
    """
    prompt = (
        f"In clinical practice, the critical low threshold for {param_name} "
        f"that requires immediate medical attention is"
    )
    try:
        completion = _generate(prompt)
        value      = _extract_first_number(completion)
        confidence = _compute_confidence(completion, unit)

        if value is None or confidence < CLINICAL_CONFIDENCE_MIN:
            logger.info(f"[ClinicalRetriever] critical_low for '{param_name}' "
                        f"unresolvable. completion='{completion}' conf={confidence:.2f}")
            return None, 0.0

        logger.info(f"[ClinicalRetriever] critical_low for '{param_name}' = {value} "
                    f"(conf={confidence:.2f}, completion='{completion}')")
        return value, confidence

    except Exception as e:
        logger.warning(f"[ClinicalRetriever] BioGPT error for '{param_name}': {e}")
        return None, 0.0


def get_critical_high(param_name: str, unit: str | None = None) -> tuple[float | None, float]:
    """Ask BioGPT: what is the critical high threshold for this clinical parameter?"""
    prompt = (
        f"In clinical practice, the dangerously high threshold for {param_name} "
        f"that requires immediate medical intervention is"
    )
    try:
        completion = _generate(prompt)
        value      = _extract_first_number(completion)
        confidence = _compute_confidence(completion, unit)

        if value is None or confidence < CLINICAL_CONFIDENCE_MIN:
            logger.info(f"[ClinicalRetriever] critical_high for '{param_name}' unresolvable.")
            return None, 0.0

        return value, confidence

    except Exception as e:
        logger.warning(f"[ClinicalRetriever] BioGPT error for '{param_name}': {e}")
        return None, 0.0


def get_normal_range(param_name: str, unit: str | None = None) -> tuple[float | None, float | None, float]:
    """
    Ask BioGPT for the normal low and high range.

    Returns:
        (normal_low, normal_high, confidence) — values may be None
    """
    prompt = (
        f"The normal reference range for {param_name} in healthy adults is"
    )
    try:
        completion = _generate(prompt)
        numbers    = re.findall(r'\b(\d+\.?\d*)\b', completion)
        confidence = _compute_confidence(completion, unit)

        if len(numbers) >= 2 and confidence >= CLINICAL_CONFIDENCE_MIN:
            low  = float(numbers[0])
            high = float(numbers[1])
            # Basic sanity: low must be less than high
            if low < high:
                return low, high, confidence

        if len(numbers) == 1 and confidence >= CLINICAL_CONFIDENCE_MIN:
            return float(numbers[0]), None, confidence

        return None, None, 0.0

    except Exception as e:
        logger.warning(f"[ClinicalRetriever] BioGPT normal range error for '{param_name}': {e}")
        return None, None, 0.0


def get_param_unit(param_name: str) -> str | None:
    """Ask BioGPT what unit is used to measure this parameter."""
    prompt = f"The standard unit of measurement for {param_name} in clinical settings is"
    try:
        completion = _generate(prompt)
        # Common unit patterns
        unit_pattern = re.compile(
            r'\b(%|bpm|mmHg|mg/dL|breaths/min|°C|Celsius|beats per minute|'
            r'millimeters of mercury|milligrams per deciliter)\b',
            re.IGNORECASE,
        )
        m = unit_pattern.search(completion)
        return m.group(0) if m else None
    except Exception:
        return None