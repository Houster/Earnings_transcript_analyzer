# lm_scorer.py
# ------------------------------------------------------------
# Rule-based sentiment scoring using the Loughran-McDonald
# Master Dictionary. Produces a second opinion alongside
# Claude's holistic scores, making your analysis more defensible.
#
# Setup (one-time):
#   1. Download the master CSV from https://sraf.nd.edu/loughranmcdonald/resources/
#   2. Save it as:  lm_dictionary/LM_MasterDictionary.csv
# ------------------------------------------------------------

import os
import re
import csv

LM_DICT_PATH = os.path.join("lm_dictionary", "LM_MasterDictionary.csv")

SENTIMENT_COLUMNS = {
    "negative":     "Negative",
    "positive":     "Positive",
    "uncertainty":  "Uncertainty",
    "litigious":    "Litigious",
    "strong_modal": "Strong_Modal",
    "weak_modal":   "Weak_Modal",
}

_lm_cache = None


def load_lm_dictionary() -> dict:
    """
    Load the LM master CSV into memory.
    Returns { "DECLINE": {"negative": True, "positive": False, ...}, ... }
    Cached after first call so subsequent calls are instant.
    """
    global _lm_cache
    if _lm_cache is not None:
        return _lm_cache

    if not os.path.exists(LM_DICT_PATH):
        raise FileNotFoundError(
            f"LM dictionary not found at '{LM_DICT_PATH}'.\n"
            "Download it from https://sraf.nd.edu/loughranmcdonald/resources/ "
            "and save it as lm_dictionary/LM_MasterDictionary.csv"
        )

    word_map = {}
    with open(LM_DICT_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].strip().upper()
            sentiments = {
                key: (row.get(col_name, "0").strip() not in ("0", "", "FALSE"))
                for key, col_name in SENTIMENT_COLUMNS.items()
            }
            if any(sentiments.values()):
                word_map[word] = sentiments

    _lm_cache = word_map
    print(f"[lm_scorer] Loaded {len(word_map):,} sentiment words.")
    return _lm_cache


def _tokenise(text: str) -> list:
    return re.findall(r"[A-Za-z]+", text.upper())


def score_text(text: str) -> dict:
    """
    Score a block of text. Returns raw counts, normalised frequencies,
    and two derived scores on your existing scales:
      tone_lm       (0 to 100)
      confidence_lm (0 to 100)
    """
    lm     = load_lm_dictionary()
    tokens = _tokenise(text)
    total  = len(tokens)

    if total == 0:
        return _empty_scores()

    counts = {key: 0 for key in SENTIMENT_COLUMNS}
    for token in tokens:
        if token in lm:
            for key, flagged in lm[token].items():
                if flagged:
                    counts[key] += 1

    freqs = {f"{key}_freq": round(counts[key] / total, 6) for key in counts}

    # Confidence: strong modal words raise it; weak modal + uncertainty lower it.
    net_confidence = (
        freqs["strong_modal_freq"]
        - freqs["weak_modal_freq"]
        - freqs["uncertainty_freq"]
    )
    confidence_lm = round(max(0.0, min(100.0, 50.0 + net_confidence * 50.0)), 2)

    # Tone: positive minus negative frequency, scaled to -100/+100.
    net_tone = freqs["positive_freq"] - freqs["negative_freq"]
    tone_lm  = round(max(0.0, min(100.0, net_tone * 100.0)), 2)

    return {
        "lm_negative_count":     counts["negative"],
        "lm_positive_count":     counts["positive"],
        "lm_uncertainty_count":  counts["uncertainty"],
        "lm_strong_modal_count": counts["strong_modal"],
        "lm_weak_modal_count":   counts["weak_modal"],
        **freqs,
        "total_words":   total,
        "tone_lm":       tone_lm,
        "confidence_lm": confidence_lm,
    }


def _empty_scores() -> dict:
    return {
        "lm_negative_count": 0, "lm_positive_count": 0,
        "lm_uncertainty_count": 0, "lm_strong_modal_count": 0,
        "lm_weak_modal_count": 0,
        "negative_freq": 0.0, "positive_freq": 0.0,
        "uncertainty_freq": 0.0, "strong_modal_freq": 0.0,
        "weak_modal_freq": 0.0, "litigious_freq": 0.0,
        "total_words": 0, "tone_lm": 0.0, "confidence_lm": 0.0,
    }


def score_parsed_transcript(parsed: dict) -> dict:
    """
    Score prepared remarks, Q&A questions, management responses, and full text
    separately. Returns a flat dict with section-prefixed keys, e.g.:
      prepared__tone_lm, qa_responses__confidence_lm, full__lm_negative_count
    """
    sections = {
        "prepared":     parsed.get("prepared_remarks", ""),
        "qa_questions": " ".join(parsed.get("analyst_questions", [])),
        "qa_responses": " ".join(parsed.get("management_responses", [])),
        "full":         parsed.get("full_text", ""),
    }

    result = {}
    for section_name, text in sections.items():
        for key, value in score_text(text).items():
            result[f"{section_name}__{key}"] = value

    return result
