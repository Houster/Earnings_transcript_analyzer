# analyzer.py
# ------------------------------------------------------------
# Pipeline:
#   1. build_analysis_prompt()    – formats transcript for Claude
#   2. analyze_transcript()       – Claude scoring
#   3. classify_industry()        – one-time industry classification per ticker
#   4. detect_changes()           – quarter-over-quarter delta analysis
#   5. run_analysis_multi_year()  – full orchestrator
# ------------------------------------------------------------

import os
import re
import json
import csv
import anthropic
from datetime import datetime

#from config            import ANTHROPIC_API_KEY, CSV_OUTPUT_PATH
from fetcher           import fetch_transcripts_multi_year
from transcript_parser import parse_transcript
from lm_scorer         import score_parsed_transcript


RESULTS_DIR = "results"

# Industry classification is cached here so we only call the API once per
# ticker per process, not once per quarter.
_INDUSTRY_CACHE: dict[str, str] = {}

# Canonical industry list — Claude is constrained to these values so the
# chatbot filter works reliably with no free-text mismatch.
INDUSTRIES = [
    "Technology",
    "Semiconductors",
    "Financials & Banking",
    "Healthcare & Biotech",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy & Utilities",
    "Industrials",
    "Real Estate",
    "Materials & Mining",
    "Telecommunications",
    "Media & Entertainment",
]

CSV_COLUMNS = [
    # Identity
    "ticker", "company_name", "date", "year", "quarter",
    # Classification
    "industry",
    # Claude scores
    "management_prepared_remarks_sentiment", "management_responses_sentiment",
    "management_overall_confidence", "analyst_pushback_level", "forward_guidance_sentiment",
    "key_themes", "notable_phrases", "summary",
    # Change detection vs. prior quarter
    "delta_management_prepared_remarks_sentiment", "delta_management_responses_sentiment",
    "delta_management_overall_confidence", "delta_forward_guidance", "delta_analyst_pushback",
    "change_flags",      # human-readable list of notable changes
    "change_summary",    # one-sentence narrative of what shifted
    # Meta
    "analyzed_at",
]


CANONICAL_THEMES = [
    "cloud growth", "AI / copilot", "digital transformation", "security",
    "margins / efficiency", "revenue growth", "guidance / outlook",
    "macro / headwinds", "supply chain", "developer platform",
    "partner ecosystem", "hardware / devices", "international expansion",
    "capex / investment", "consumption model", "M&A / integration",]
    # extend as needed — these are short, cross-industry, reusable

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_prompt(parsed: dict) -> str:
    MAX_CHARS = 8_000
    prepared  = parsed["prepared_remarks"][:MAX_CHARS]
    questions = "\n\n".join(parsed["analyst_questions"])[:MAX_CHARS]
    responses = "\n\n".join(parsed["management_responses"])[:MAX_CHARS]
    themes_list = "\n".join(f"- {t}" for t in CANONICAL_THEMES)
    LM_results = score_parsed_transcript(parsed)

    return f"""
You are a senior equity analyst specialising in qualitative earnings-call analysis.
Analyse the transcript below and return ONLY a valid JSON object — no markdown fences,
no explanation, no preamble. Rule-based Loughran-McDonald dictionary scores for this
transcript are included for your reference (scored 0-100).

=== COMPANY ===
{parsed['company_name']}  |  Date: {parsed['date']}

=== PREPARED REMARKS ===
{prepared}
LM scores (prepared remarks): tone={LM_results.get('prepared__tone_lm', 'N/A')}, confidence={LM_results.get('prepared__confidence_lm', 'N/A')}, uncertainty_freq={LM_results.get('prepared__uncertainty_freq', 'N/A')}

=== ANALYST QUESTIONS ===
{questions}
LM scores (analyst questions): tone={LM_results.get('qa_questions__tone_lm', 'N/A')}, confidence={LM_results.get('qa_questions__confidence_lm', 'N/A')}, uncertainty_freq={LM_results.get('qa_questions__uncertainty_freq', 'N/A')}

=== MANAGEMENT RESPONSES ===
{responses}
LM scores (management responses): tone={LM_results.get('qa_responses__tone_lm', 'N/A')}, confidence={LM_results.get('qa_responses__confidence_lm', 'N/A')}, uncertainty_freq={LM_results.get('qa_responses__uncertainty_freq', 'N/A')}

Return a JSON object with EXACTLY these keys:
{{
  "management_prepared_remarks_sentiment": <integer 0 to 100>,
  "management_responses_sentiment":        <integer 0 to 100>,
  "management_overall_confidence":         <integer 0 to 100>,
  "analyst_pushback_level":                <integer 0 to 100>,
  "forward_guidance_sentiment":            <integer 0 to 100>,
  "key_themes":                            key_themes: choose 3 to 5 from the list below. If none fit, create a new label of 2 to 3 words maximum — never more than 4 words. Valid themes (prefer these): {themes_list}
  "notable_phrases":                       [<string>, ...],
  "summary":                               "<one sentence>"
}}

Scoring:
- management_prepared_remarks_sentiment:  0=very negative, 50=neutral, 100=very positive
- management_responses_sentiment:         0=very negative, 50=neutral, 100=very positive
- management_overall_confidence:          0=heavily hedged, 50=neutral, 100=highly assertive
- analyst_pushback_level:                 0=softball, 50=neutral, 100=aggressive scrutiny
- forward_guidance_sentiment:             0=guiding down, 50=neutral, 100=guiding up
- key_themes:                             3–5 dominant topics
- notable_phrases:                        2–3 exact phrases that drove your scoring
- Base all scores on language, not reported numbers.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Claude API call — transcript scoring
# ─────────────────────────────────────────────────────────────────────────────

def analyze_transcript(parsed: dict) -> dict:
    client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model       = "claude-sonnet-4-5",
        max_tokens  = 1024,
        temperature = 0,        # deterministic — same input always gives same output
        messages    = [{"role": "user", "content": build_analysis_prompt(parsed)}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$",          "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON: {e}\n\nRaw:\n{raw}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Industry classification  (one call per ticker, cached in memory + file)
# ─────────────────────────────────────────────────────────────────────────────

def _industry_cache_path() -> str:
    return os.path.join(RESULTS_DIR, "_industry_cache.json")


def _load_industry_file_cache() -> dict[str, str]:
    path = _industry_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_industry_file_cache(cache: dict[str, str]) -> None:
    path = _industry_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def classify_industry(ticker: str, company_name: str) -> str:
    """
    Return the industry for a ticker using a three-layer cache:
      1. In-memory dict  (_INDUSTRY_CACHE) — free, instant
      2. File cache      (results/_industry_cache.json) — survives restarts
      3. Claude Haiku API call — only on a true cache miss

    Always returns one of the strings in INDUSTRIES.
    """
    global _INDUSTRY_CACHE
    ticker_upper = ticker.upper()

    if ticker_upper in _INDUSTRY_CACHE:
        return _INDUSTRY_CACHE[ticker_upper]

    file_cache = _load_industry_file_cache()
    if ticker_upper in file_cache:
        _INDUSTRY_CACHE[ticker_upper] = file_cache[ticker_upper]
        return file_cache[ticker_upper]

    print(f"[analyzer] Classifying industry for {ticker_upper} ({company_name}) ...")
    industries_str = "\n".join(f"- {ind}" for ind in INDUSTRIES)
    prompt = (
        f"You are a financial data classifier.\n\n"
        f"Classify the company below into exactly ONE of the following industry categories.\n"
        f"Return ONLY the category name — no explanation, no punctuation, nothing else.\n\n"
        f"Company: {company_name} (ticker: {ticker_upper})\n\n"
        f"Valid categories:\n{industries_str}"
    )

    try:
        client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model       = "claude-haiku-4-5-20251001",
            max_tokens  = 16,
            temperature = 0,
            messages    = [{"role": "user", "content": prompt}],
        )
        result = message.content[0].text.strip()

        if result not in INDUSTRIES:
            match  = next((ind for ind in INDUSTRIES if ind.lower() == result.lower()), None)
            result = match if match else "Technology"
    except Exception as e:
        print(f"[analyzer] Industry classification failed for {ticker_upper}: {e} — defaulting.")
        result = "Technology"

    _INDUSTRY_CACHE[ticker_upper] = result
    file_cache[ticker_upper]      = result
    _save_industry_file_cache(file_cache)
    print(f"[analyzer] {ticker_upper} -> {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Change detection
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_THRESHOLDS = {
    "management_prepared_remarks_sentiment": 10.0,
    "management_responses_sentiment":        10.0,
    "management_overall_confidence":         10.0,
    "forward_guidance_sentiment":            10.0,
    "analyst_pushback_level":                10.0,
}

FIELD_LABELS = {
    "management_prepared_remarks_sentiment": "Prepared remarks sentiment",
    "management_responses_sentiment":        "Management responses sentiment",
    "management_overall_confidence":         "Management overall confidence",
    "forward_guidance_sentiment":            "Forward guidance sentiment",
    "analyst_pushback_level":                "Analyst pushback level",
}


def detect_changes(current: dict, prior: dict | None) -> dict:
    """
    Compare current quarter's scores to the prior quarter.

    Returns dict with delta_* fields, change_flags, and change_summary.
    """
    if prior is None:
        return {
            "delta_management_prepared_remarks_sentiment": None,
            "delta_management_responses_sentiment":        None,
            "delta_management_overall_confidence":         None,
            "delta_forward_guidance":                      None,
            "delta_analyst_pushback":                      None,
            "change_flags":                                "first quarter — no prior period",
            "change_summary":                              "No prior quarter available for comparison.",
        }

    def safe_delta(key: str) -> float | None:
        try:
            return round(float(current[key]) - float(prior[key]), 4)
        except (TypeError, ValueError, KeyError):
            return None

    deltas = {
        "delta_management_prepared_remarks_sentiment": safe_delta("management_prepared_remarks_sentiment"),
        "delta_management_responses_sentiment":        safe_delta("management_responses_sentiment"),
        "delta_management_overall_confidence":         safe_delta("management_overall_confidence"),
        "delta_forward_guidance":                      safe_delta("forward_guidance_sentiment"),
        "delta_analyst_pushback":                      safe_delta("analyst_pushback_level"),
    }

    delta_to_field = {
        "delta_management_prepared_remarks_sentiment": "management_prepared_remarks_sentiment",
        "delta_management_responses_sentiment":        "management_responses_sentiment",
        "delta_management_overall_confidence":         "management_overall_confidence",
        "delta_forward_guidance":                      "forward_guidance_sentiment",
        "delta_analyst_pushback":                      "analyst_pushback_level",
    }

    flags = []
    for delta_key, field in delta_to_field.items():
        value     = deltas.get(delta_key)
        threshold = CHANGE_THRESHOLDS.get(field, 0.5)
        if value is not None and abs(value) >= threshold:
            direction = "increased" if value > 0 else "decreased"
            flags.append(f"{FIELD_LABELS.get(field, field)} {direction} by {abs(value):.2f}")

    prior_label = f"{prior.get('year','?')} Q{prior.get('quarter','?')}"

    change_summary = (
        f"No material changes detected vs {prior_label}; scores broadly stable QoQ."
        if not flags else
        f"vs {prior_label}: {'; '.join(flags[:3])}."
    )

    return {
        **deltas,
        "change_flags":   "; ".join(flags) if flags else "no material changes",
        "change_summary": change_summary,
    }


def _load_existing_analysis_keys() -> set[tuple[str, str, str]]:
    """Return existing CSV keys by (company_name, year, quarter)."""
    existing: set[tuple[str, str, str]] = set()
    if not os.path.exists(CSV_OUTPUT_PATH):
        return existing
    with open(CSV_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing.add((
                row.get("company_name", "").strip(),
                row.get("year",         "").strip(),
                row.get("quarter",      "").strip(),
            ))
    return existing


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_multi_year(
    ticker: str,
    years:  int  = 5,
    force_download: bool = False,
) -> list[dict]:
    """
    Full pipeline: ticker + years -> per-quarter JSON files + master CSV.

    For each quarter:
      1. Fetch via EarningsCall API (speaker-block format)
      2. Parse into 6-field structure
      3. Classify industry  (cached — at most one API call per ticker ever)
      4. Claude scoring     (temperature=0 for reproducibility)
      5. Change detection vs prior quarter
      6. Save JSON + append to CSV
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    existing_keys   = _load_existing_analysis_keys()
    raw_transcripts = fetch_transcripts_multi_year(
        ticker, years=years, force_download=force_download
    )

    if not raw_transcripts:
        raise RuntimeError(f"No transcripts found for {ticker}.")

    industry:     str | None = None
    all_results:  list[dict] = []
    prior_result: dict | None = None

    for raw in raw_transcripts:
        year    = raw.get("year",    "?")
        quarter = raw.get("quarter", "?")
        label   = f"{ticker.upper()} {year} Q{quarter}"

        # ── Parse ──────────────────────────────────────────────────────────
        try:
            parsed = parse_transcript(raw)
        except ValueError as e:
            print(f"[analyzer] SKIP {label} — parse error: {e}")
            continue

        # ── Industry (once per ticker) ─────────────────────────────────────
        if industry is None:
            industry = classify_industry(ticker, parsed["company_name"])

        # ── Skip if already in CSV ─────────────────────────────────────────
        csv_key = (
            parsed["company_name"].strip(),
            str(year).strip(),
            str(quarter).strip(),
        )
        if csv_key in existing_keys:
            print(f"[analyzer] SKIP {label} — already in CSV.")
            continue

        # ── Claude scoring ─────────────────────────────────────────────────
        try:
            claude_scores = analyze_transcript(parsed)
        except Exception as e:
            print(f"[analyzer] SKIP {label} — Claude error: {e}")
            continue

        # ── Assemble output record ─────────────────────────────────────────
        output: dict = {
            "ticker":       ticker.upper(),
            "company_name": parsed["company_name"],
            "date":         parsed["date"],
            "year":         year,
            "quarter":      quarter,
            "industry":     industry,          # ← was missing in prior version
            **claude_scores,
        }

        # ── Change detection ───────────────────────────────────────────────
        output.update(detect_changes(output, prior_result))

        # ── Timestamp ─────────────────────────────────────────────────────
        output["analyzed_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # ── Save JSON ──────────────────────────────────────────────────────
        json_path = os.path.join(
            RESULTS_DIR, f"{ticker.upper()}_{year}_Q{quarter}_analysis.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[analyzer] Saved -> {json_path}")

        # ── Append to CSV ──────────────────────────────────────────────────
        _append_to_csv(output)

        all_results.append(output)
        prior_result = output

    print(f"\n[analyzer] Done. {len(all_results)} quarters analysed for {ticker.upper()}.\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_to_csv(output: dict) -> None:
    """Append one result row to the master CSV. Skips exact duplicates."""
    existing_keys: set[tuple[str, str]] = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        with open(CSV_OUTPUT_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row.get("ticker", ""), row.get("date", "")))

    key = (output.get("ticker", ""), output.get("date", ""))
    if key in existing_keys:
        print(f"[analyzer] CSV: {key} already present — skipping.")
        return

    file_exists = os.path.exists(CSV_OUTPUT_PATH)
    row = {col: output.get(col, "") for col in CSV_COLUMNS}

    for list_col in ("key_themes", "notable_phrases"):
        val = row.get(list_col, "")
        if isinstance(val, list):
            row[list_col] = "; ".join(val)

    with open(CSV_OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[analyzer] CSV updated -> {CSV_OUTPUT_PATH}")