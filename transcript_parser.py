# transcript_parser.py
# ------------------------------------------------------------
# Parses earnings call transcripts from the EarningsCall API
# (https://earningscall.biz/) via fetcher.py.
#
# The fetcher now returns pre-split structured speaker blocks rather
# than a raw text string, so this parser no longer needs to detect
# the Q&A boundary or parse speaker turns from text — both are done
# upstream in fetcher.py.
#
# Input — raw dict from fetcher.fetch_transcripts_multi_year():
#   {
#     'ticker':                   'AAPL',
#     'company_name':             'Apple Inc.',
#     'date':                     '2024-08-15',
#     'year':                     2024,
#     'quarter':                  3,
#     'speaker_blocks_before_qa': [
#         {'speaker_name': 'Tim Cook', 'speaker_title': 'CEO', 'text': '...'},
#         ...
#     ],
#     'speaker_blocks_after_qa': [
#         {'speaker_name': 'Amit Daryanani', 'speaker_title': 'Evercore ISI', 'text': '...'},
#         {'speaker_name': 'Tim Cook',        'speaker_title': 'CEO',          'text': '...'},
#         ...
#     ],
#   }
#
# Output — 6-field dict (interface unchanged — analyzer.py unaffected):
#   company_name, date, prepared_remarks,
#   analyst_questions, management_responses, full_text
# ------------------------------------------------------------

import re
from datetime import datetime


# ══════════════════════════════════════════════════════════════════
# 1.  SPEAKER ROLE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

_MGMT_ROLE_RE = re.compile(
    r"chairman|vice\s+chair(?:man)?|"
    r"chief\s+(?:executive|financial|operating|accounting|"
    r"revenue|strategy|technology|information|marketing|"
    r"commercial|legal|product|people|human\s+resources)\s+officer|"
    r"\bceo\b|\bcfo\b|\bcoo\b|\bcto\b|\bcmo\b|\bclo\b|"
    r"president|"
    r"general\s+(?:manager|counsel)|"
    r"investor\s+relations|"
    r"(?:executive|senior|group)\s+vice\s+president|"
    r"\bevp\b|\bsvp\b|\bvp\b(?:\s+of)?|"
    r"(?:head|director)\s+of\s+(?:finance|ir|investor|strategy)",
    re.IGNORECASE,
)

_ANALYST_ROLE_RE = re.compile(
    r"\banalyst\b|research\s+(?:analyst|division)|"
    r"securities|capital\s+markets|"
    r"partners|bancorp|\bbank\b|"
    r"\bllc\b|\bllp\b|\binc\b|"
    r"associates|advisors|"
    r"cowen|barclays|morgan\s+stanley|goldman|jpmorgan|jp\s+morgan|"
    r"wells\s+fargo|scotiabank|evercore|bernstein|"
    r"citigroup|\bciti\b|\brbc\b|\bubs\b|piper\s+sandler|"
    r"wolfe\s+research|bmo\s+capital|\bbofa\b|bank\s+of\s+america|"
    r"deutsche\s+bank|credit\s+suisse|macquarie|raymond\s+james|"
    r"stifel|truist|baird|needham|oppenheimer|mizuho|jefferies|"
    r"guggenheim|keybanc|loop\s+capital|d\.a\.\s+davidson|"
    r"william\s+blair|canaccord|rosenblatt|susquehanna|\bpiper\b",
    re.IGNORECASE,
)


def _classify_role(speaker_name: str, speaker_title: str) -> str:
    """
    Return 'operator' | 'management' | 'analyst' | 'unknown'.

    Priority:
      1. Name is literally 'Operator' / 'Conference Operator' / 'Moderator'
      2. Title matches management regex
      3. Title matches analyst regex
      4. Name matches analyst firm regex  (some transcripts omit firm in title)
      5. Unknown — caller handles as needed
    """
    name_lower = speaker_name.strip().lower()
    if name_lower in ("operator", "conference operator", "moderator"):
        return "operator"

    if speaker_title:
        if _MGMT_ROLE_RE.search(speaker_title):
            return "management"
        if _ANALYST_ROLE_RE.search(speaker_title):
            return "analyst"

    # Fallback: check name itself for analyst firm keywords
    if _ANALYST_ROLE_RE.search(speaker_name):
        return "analyst"

    return "unknown"


# ══════════════════════════════════════════════════════════════════
# 2.  PREPARED REMARKS ASSEMBLY
# ══════════════════════════════════════════════════════════════════

_SECTION_NOISE_RE = re.compile(
    r"^(?:presentation|prepared\s+remarks?|earnings\s+call\s+transcript)\s*$",
    re.IGNORECASE,
)


def _build_prepared_remarks(blocks: list[dict]) -> str:
    """
    Concatenate all before-Q&A speaker blocks into a single prose string.

    Operator logistics blocks are dropped. Management and unknown speaker
    blocks are included with "Speaker Name: <text>" formatting so Claude
    has attribution context when scoring.
    """
    parts = []
    for block in blocks:
        name  = block.get("speaker_name", "")
        title = block.get("speaker_title", "")
        text  = (block.get("text") or "").strip()

        if not text or _SECTION_NOISE_RE.match(text):
            continue

        if _classify_role(name, title) == "operator":
            continue

        parts.append(f"{name}: {text}" if name else text)

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# 3.  Q&A BLOCK CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def _parse_qa_blocks(blocks: list[dict]) -> tuple[list[str], list[str]]:
    """
    Separate after-Q&A speaker blocks into paired analyst questions and
    management responses.

    Multiple consecutive management turns for the same question are
    joined into a single response string so the lists stay equal-length
    and properly paired.

    Returns
    -------
    analyst_questions    : list[str]
    management_responses : list[str]
    """
    analyst_questions:   list[str] = []
    management_responses: list[str] = []

    current_question:  str       = ""
    current_responses: list[str] = []

    def _flush() -> None:
        nonlocal current_question, current_responses
        if current_question:
            analyst_questions.append(current_question)
            management_responses.append(" ".join(current_responses))
        current_question  = ""
        current_responses = []

    for block in blocks:
        name  = block.get("speaker_name", "")
        title = block.get("speaker_title", "")
        text  = (block.get("text") or "").strip()

        if not text:
            continue

        role = _classify_role(name, title)

        if role == "operator":
            # Operator introduces the next questioner — skip, don't flush
            continue

        elif role == "analyst":
            _flush()                     # save previous Q&A pair
            current_question  = text
            current_responses = []

        elif role == "management":
            current_responses.append(text)

        else:
            # Unknown: extend current response if question is open,
            # otherwise treat as a new question
            if current_question:
                current_responses.append(text)
            else:
                current_question = text

    _flush()  # capture the final pair

    # ── Fallback: no roles resolved (transcript has no title metadata) ────
    if not analyst_questions and blocks:
        all_texts = [
            (b.get("text") or "").strip()
            for b in blocks
            if (b.get("text") or "").strip()
        ]
        analyst_questions    = all_texts[0::2]
        management_responses = all_texts[1::2]

        max_len = max(len(analyst_questions), len(management_responses))
        analyst_questions    += [""] * (max_len - len(analyst_questions))
        management_responses += [""] * (max_len - len(management_responses))

    return analyst_questions, management_responses


# ══════════════════════════════════════════════════════════════════
# 4.  FULL-TEXT ASSEMBLY (for lm_scorer)
# ══════════════════════════════════════════════════════════════════

def _build_full_text(before_blocks: list[dict], after_blocks: list[dict]) -> str:
    """
    Reconstruct a single readable transcript string from all speaker blocks.
    lm_scorer expects a flat full_text field; this provides it.
    Format: "Speaker Name: <text>" per block, blank line between turns.
    """
    lines = []
    for block in before_blocks + after_blocks:
        name = block.get("speaker_name", "")
        text = (block.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{name}: {text}" if name else text)
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 5.  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def parse_transcript(raw: dict) -> dict:
    """
    Convert a raw fetcher dict (speaker-block format) into the 6-field
    structure expected by analyzer.py and lm_scorer.py.

    Parameters
    ----------
    raw : dict
        Output of fetcher.fetch_transcripts_multi_year(). Must contain:
          'ticker', 'company_name', 'date', 'year', 'quarter',
          'speaker_blocks_before_qa', 'speaker_blocks_after_qa'

    Returns
    -------
    dict with keys:
        company_name         : str
        date                 : str  (ISO YYYY-MM-DD)
        prepared_remarks     : str
        analyst_questions    : list[str]
        management_responses : list[str]
        full_text            : str

    Raises
    ------
    ValueError
        If the raw dict contains no speaker blocks at all (fetcher failure).
    """
    # ── Metadata ──────────────────────────────────────────────────────────
    company_name = (
        raw.get("company_name")
        or raw.get("companyName")
        or raw.get("company")
        or "Unknown Company"
    ).strip()

    raw_date = raw.get("date") or ""
    try:
        date = datetime.strptime(raw_date[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        date = raw_date or "unknown"

    # ── Speaker blocks (already split by fetcher) ─────────────────────────
    before_blocks = raw.get("speaker_blocks_before_qa") or []
    after_blocks  = raw.get("speaker_blocks_after_qa")  or []

    if not before_blocks and not after_blocks:
        raise ValueError(
            f"No speaker blocks found for {raw.get('ticker','?')} "
            f"{raw.get('year','?')} Q{raw.get('quarter','?')}. "
            "Fetcher returned empty data."
        )

    # ── Assemble output fields ─────────────────────────────────────────────
    prepared_remarks                       = _build_prepared_remarks(before_blocks)
    analyst_questions, management_responses = _parse_qa_blocks(after_blocks)
    full_text                              = _build_full_text(before_blocks, after_blocks)

    # ── Warn on empty sections (don't crash — partial data is still useful) ─
    ticker_label = f"{raw.get('ticker','?')} {raw.get('year','?')} Q{raw.get('quarter','?')}"
    if not prepared_remarks:
        print(f"[parser] WARNING: No prepared remarks extracted for {ticker_label}.")
    if not analyst_questions:
        print(f"[parser] WARNING: No analyst questions extracted for {ticker_label}.")

    return {
        "company_name":          company_name,
        "date":                  date,
        "prepared_remarks":      prepared_remarks,
        "analyst_questions":     analyst_questions,
        "management_responses":  management_responses,
        "full_text":             full_text,
    }