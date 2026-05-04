"""
Sentiment analysis module using FinBERT for financial text.
"""
import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512                # FinBERT's context window limit
POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"
NEUTRAL_LABEL = "neutral"

# Threshold for labelling the sentiment
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1


def _load_finbert():
    """
    Lazy-load FinBERT tokenizer and model.
    Downloads on first call, then served from HuggingFace local cache.

    Returns
    -------
    Tuple[AutoTokenizer, AutoModelForSequenceClassification]
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading FinBERT model (%s)…", FINBERT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()
    logger.info("FinBERT loaded successfully")
    return tokenizer, model


# Module-level cache so the model is only loaded once per Python session
_tokenizer = None
_model = None


def _get_finbert():
    """Return cached FinBERT instances, loading them on first call."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer, _model = _load_finbert()
    return _tokenizer, _model


def score_text(text: str) -> Tuple[float, str]:
    """
    Score financial text using FinBERT and return a scalar sentiment score.

    The scalar score is computed as:
        score = P(positive) - P(negative)    ∈ [-1.0, +1.0]

    Where:
        -1.0 = strongly negative
         0.0 = neutral
        +1.0 = strongly positive

    Text is truncated to MAX_TOKENS before scoring if longer.

    Parameters
    ----------
    text : str
        Raw article text to score.

    Returns
    -------
    Tuple[float, str]
        (sentiment_score, label) where label ∈ {'Positive', 'Negative', 'Neutral'}
    """
    if not text or not text.strip():
        logger.info("Empty text received — returning neutral sentiment (0.0)")
        return 0.0, NEUTRAL_LABEL.capitalize()

    tokenizer, model = _get_finbert()

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Softmax to convert logits to class probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

        # FinBERT label order: positive=0, negative=1, neutral=2
        # (verified from model card — label2id mapping)
        label_map = {v: k for k, v in model.config.label2id.items()}
        prob_dict = {label_map[i]: probs[i].item() for i in range(len(probs))}

        p_positive = prob_dict.get(POSITIVE_LABEL, 0.0)
        p_negative = prob_dict.get(NEGATIVE_LABEL, 0.0)

        score = float(p_positive - p_negative)
        label = _score_to_label(score)

        logger.info(
            "Sentiment scored: %.4f (%s) — P(pos)=%.3f, P(neg)=%.3f, P(neu)=%.3f",
            score, label,
            prob_dict.get(POSITIVE_LABEL, 0),
            prob_dict.get(NEGATIVE_LABEL, 0),
            prob_dict.get(NEUTRAL_LABEL, 0),
        )
        return score, label

    except Exception as exc:
        logger.error("FinBERT scoring failed: %s. Returning neutral.", exc)
        return 0.0, NEUTRAL_LABEL.capitalize()


def _score_to_label(score: float) -> str:
    """Map a scalar score in [-1, 1] to a human-readable label."""
    if score > POSITIVE_THRESHOLD:
        return "Positive"
    if score < NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a PDF file using PyMuPDF (fitz).

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content (e.g. from Streamlit file_uploader.read()).

    Returns
    -------
    str
        Concatenated text content of all pages.

    Raises
    ------
    RuntimeError
        If PyMuPDF fails to open or parse the PDF.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
        full_text = "\n".join(pages_text)
        logger.info("Extracted %d characters from PDF (%d pages)", len(full_text), len(pages_text))
        return full_text
    except Exception as exc:
        raise RuntimeError(f"PDF text extraction failed: {exc}") from exc


def score_article(
    text: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
) -> Tuple[float, str, str]:
    """
    High-level entry point: score an article provided as text or PDF bytes.

    Exactly one of `text` or `pdf_bytes` should be provided. If neither is
    provided, returns a neutral score of 0.0.

    Parameters
    ----------
    text : str, optional
        Plain text of the article.
    pdf_bytes : bytes, optional
        Raw bytes of a PDF file.

    Returns
    -------
    Tuple[float, str, str]
        (score, label, snippet) where:
        - score ∈ [-1.0, +1.0]
        - label ∈ {'Positive', 'Neutral', 'Negative'}
        - snippet is the first 300 characters of the article (for display)
    """
    if pdf_bytes is not None:
        article_text = extract_text_from_pdf(pdf_bytes)
    elif text:
        article_text = text
    else:
        logger.info("No article provided — using neutral sentiment score")
        return 0.0, "Neutral", ""

    snippet = article_text[:300].strip()
    score, label = score_text(article_text)
    return score, label, snippet
