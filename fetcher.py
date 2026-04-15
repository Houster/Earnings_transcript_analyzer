# fetcher.py
# ------------------------------------------------------------
# Fetches earnings call transcripts using the EarningsCall API
# (https://earningscall.biz/).
#
# This module returns structured transcript data with speaker blocks
# split into pre-Q&A and Q&A sections.
#
# SETUP:
#   pip install earningscall
#
# API KEY (optional but recommended):
#   Set environment variable: export EARNINGSCALL_API_KEY="your-key"
#   Or programmatically: earningscall.api_key = "your-key"
# ------------------------------------------------------------

import re
from datetime import date, datetime

try:
    from earningscall import get_company
    from requests.exceptions import HTTPError
    API_SUPPORT = True
except ImportError:
    API_SUPPORT = False
    print(
        "[fetcher] WARNING: earningscall not installed. API fetching disabled.\n"
        "          Fix with:  pip install earningscall"
    )

# Earnings calls happen ~4-6 weeks after quarter end.
# We approximate the call date as mid-month of the reporting month.
#   Q1 (Jan-Mar) -> reported April   -> use April 15
#   Q2 (Apr-Jun) -> reported July    -> use July 15
#   Q3 (Jul-Sep) -> reported October -> use October 15
#   Q4 (Oct-Dec) -> reported January of NEXT year -> use January 15
QUARTER_TO_MONTH = {1: 4, 2: 7, 3: 10, 4: 1}

_ANALYST_ROLE_RE = re.compile(
    r"analyst|research|securities|capital\s+markets|bank|bancorp|llc|llp|inc|"
    r"associates|advisors|cowen|barclays|morgan\s+stanley|goldman|jpmorgan|jp\s+morgan|"
    r"wells\s+fargo|scotiabank|evercore|bernstein|citigroup|citi|rbc|ubs|piper\s+sandler|"
    r"wolfe\s+research|bmo\s+capital|deutsche\s+bank|credit\s+suisse|macquarie|raymond\s+james|"
    r"stifel|truist|baird|needham|oppenheimer|mizuho|jefferies|guggenheim|keybanc|loop\s+capital|"
    r"d\.a\.\s+davidson|william\s+blair|canaccord|rosenblatt|susquehanna|piper",
    re.IGNORECASE,
)

_QA_BOUNDARY_PATTERNS = [
    re.compile(r"our first question", re.IGNORECASE),
    re.compile(r"first question.*from", re.IGNORECASE),
    re.compile(r"question comes from", re.IGNORECASE),
    re.compile(r"we(?:'ll| will) (?:go ahead and )?take (?:our )?first question", re.IGNORECASE),
    re.compile(r"may we have the first question", re.IGNORECASE),
]



# =============================================================================
# API fetching (EarningsCall)
# =============================================================================

def _infer_date(year: int, quarter: int) -> str:
    month = QUARTER_TO_MONTH[quarter]
    report_year = year + 1 if quarter == 4 else year
    return date(report_year, month, 15).isoformat()


def _extract_date_from_text(text: str) -> str | None:
    header = "\n".join(text.splitlines()[:10])
    candidates = [
        (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{1,2}/\d{1,2}/\d{4})\b", "%m/%d/%Y"),
        (r"\b((?:January|February|March|April|May|June|July|"
         r"August|September|October|November|December)\.?(?:\s+\d{1,2},?\s+\d{4}))\b", None),
        (r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?(?:\s+\d{1,2},?\s+\d{4}))\b", None),
    ]

    for pattern, fmt in candidates:
        m = re.search(pattern, header, re.IGNORECASE)
        if not m:
            continue
        raw = re.sub(r"\.", "", m.group(1)).strip()
        if fmt:
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
        for attempt in ("%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y"):
            try:
                cleaned = re.sub(r",", "", raw)
                return datetime.strptime(cleaned, attempt).date().isoformat()
            except ValueError:
                continue
    return None


def _is_analyst_speaker(speaker_name: str, speaker_title: str) -> bool:
    if _ANALYST_ROLE_RE.search(speaker_name):
        return True
    if _ANALYST_ROLE_RE.search(speaker_title):
        return True
    return False


def _speaker_to_dict(speaker) -> dict:
    info = getattr(speaker, "speaker_info", None)
    speaker_name = getattr(info, "name", None) if info else None
    speaker_title = getattr(info, "title", None) if info else None

    block = {
        "speaker_id": getattr(speaker, "speaker", ""),
        "speaker_name": speaker_name or str(getattr(speaker, "speaker", "")),
        "speaker_title": speaker_title or "",
        "text": (getattr(speaker, "text", "") or "").strip(),
    }
    if hasattr(speaker, "start_times") and getattr(speaker, "start_times") is not None:
        block["start_times"] = list(getattr(speaker, "start_times"))
    return block


def _find_qna_start_index(speakers: list) -> int:
    for idx, speaker in enumerate(speakers):
        text = (getattr(speaker, "text", "") or "").lower()
        for pat in _QA_BOUNDARY_PATTERNS:
            if pat.search(text):
                return idx

    for idx, speaker in enumerate(speakers):
        speaker_name = getattr(getattr(speaker, "speaker_info", None), "name", "")
        speaker_title = getattr(getattr(speaker, "speaker_info", None), "title", "")
        if _is_analyst_speaker(speaker_name, speaker_title):
            return idx

    return len(speakers)


def _get_transcript_date(transcript, year: int, quarter: int) -> str:
    if hasattr(transcript, "event") and getattr(transcript.event, "conference_date", None):
        try:
            return transcript.event.conference_date.date().isoformat()
        except Exception:
            pass
    if hasattr(transcript, "text") and transcript.text:
        return _extract_date_from_text(transcript.text) or _infer_date(year, quarter)
    return _infer_date(year, quarter)


def _fetch_from_api(ticker: str, year: int, quarter: int) -> dict | None:
    if not API_SUPPORT:
        raise RuntimeError("earningscall is not installed; cannot fetch transcripts.")

    try:
        company = get_company(ticker.upper())
        transcript = company.get_transcript(year=year, quarter=quarter, level=2)

        if not transcript or not getattr(transcript, "text", None):
            print(f"[fetcher] API: No transcript text available for {ticker.upper()} {year} Q{quarter}")
            return None

        date_str = _get_transcript_date(transcript, year, quarter)
        speakers = getattr(transcript, "speakers", None) or []
        speaker_blocks = [_speaker_to_dict(s) for s in speakers]

        if not speaker_blocks:
            speaker_blocks = [{
                "speaker_id": "unknown",
                "speaker_name": "Unknown",
                "speaker_title": "",
                "text": transcript.text.strip(),
            }]

        qna_start = _find_qna_start_index(speakers)
        before_blocks = speaker_blocks[:qna_start]
        after_blocks = speaker_blocks[qna_start:]

        print(f"[fetcher] API: Fetched {ticker.upper()}_{year}_Q{quarter} ({len(transcript.text):,} chars) with {len(before_blocks)} before-Q&A blocks and {len(after_blocks)} after-Q&A blocks")

        return {
            "ticker": ticker.upper(),
            "company_name": company.company_info.name if hasattr(company, "company_info") else ticker.upper(),
            "date": date_str,
            "year": year,
            "quarter": quarter,
            "speaker_blocks_before_qa": before_blocks,
            "speaker_blocks_after_qa": after_blocks,
        }

    except HTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 404:
            print(f"[fetcher] API: Transcript not found on API ({ticker.upper()} {year} Q{quarter})")
        else:
            print(f"[fetcher] API: HTTP {status_code} for {ticker.upper()} {year} Q{quarter}")
        return None
    except Exception as e:
        print(f"[fetcher] API: Error fetching {ticker.upper()} {year} Q{quarter}: {e}")
        return None



# =============================================================================
# Public API  (same interface as before — analyzer.py is unchanged)
# =============================================================================

def load_single_transcript(ticker: str, year: int, quarter: int) -> dict | None:
    """
    Load one transcript for a given ticker/year/quarter.
    Tries:
      1. EarningsCall API (if available)
      2. Local .pdf file
      3. Local .txt file
    Returns None if all sources fail.
    """
    # Try API first
    result = _fetch_from_api(ticker, year, quarter)
    if result:
        return result

    # Fall back to local files
    for ext in (".pdf", ".txt"):
        filename = f"{ticker.upper()}_{year}_Q{quarter}{ext}"
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        if os.path.exists(filepath):
            print(f"[fetcher] Loading local file: {filename}")
            try:
                return _load_file(filepath, ticker.upper(), year, quarter)
            except ValueError as e:
                print(f"[fetcher] WARNING: {e}")
                return None

    print(f"[fetcher] Not found: {ticker.upper()}_{year}_Q{quarter} (API or local file)")
    return None


def fetch_transcripts_multi_year(
    ticker: str,
    years: int = 5,
    force_download: bool = False,   # kept for interface compatibility
) -> list[dict]:
    """Fetch transcripts for a ticker for the last `years` years using the API only."""
    if not API_SUPPORT:
        raise RuntimeError("earningscall is not installed; cannot load transcripts.")

    cutoff_year = date.today().year - years
    current_year = date.today().year
    current_quarter = (date.today().month - 1) // 3 + 1

    results = []
    for year in range(cutoff_year, current_year + 1):
        for quarter in range(1, 5):
            if year == current_year and quarter > current_quarter:
                continue
            result = _fetch_from_api(ticker, year, quarter)
            if result:
                results.append(result)

    return results






