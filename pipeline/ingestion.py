# pipeline/ingestion.py
# Reads PDF, DOCX, or TXT files and returns a list of sentences using spaCy.

import spacy
from pathlib import Path

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def _read_pdf(file_path: str) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _read_docx(file_path: str) -> str:
    import docx
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_document(file_path: str) -> list:
    """
    Read a PDF, DOCX, or TXT file and return a list of clean sentences.

    Args:
        file_path: Path to the document.

    Returns:
        List of sentence strings, stripped and non-empty (min 10 chars).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = _read_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        raw_text = _read_docx(file_path)
    elif suffix in (".txt", ".text", ""):
        raw_text = _read_txt(file_path)
    else:
        # Attempt plain text as fallback
        raw_text = _read_txt(file_path)

    nlp = _get_nlp()
    # spaCy has a max length limit; process in chunks if needed
    max_length = 1_000_000
    if len(raw_text) > max_length:
        raw_text = raw_text[:max_length]

    doc = nlp(raw_text)
    sentences = []
    for sent in doc.sents:
        clean = sent.text.strip()
        # Remove leading numbering like "1." or "25."
        import re
        clean = re.sub(r'^\d+\.\s*', '', clean).strip()
        if len(clean) >= 10:
            sentences.append(clean)

    return sentences
