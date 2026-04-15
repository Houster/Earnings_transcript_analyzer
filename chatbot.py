# chatbot.py
# ------------------------------------------------------------
# Streamlit chatbot for earnings call tone analysis.
# Run:  streamlit run chatbot.py
# ------------------------------------------------------------

import os
import re
import anthropic
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials
from collections import Counter
from datetime import datetime
from config            import ANTHROPIC_API_KEY, CSV_OUTPUT_PATH
from main import initialize_data

# ── Config ────────────────────────────────────────────────────────────────────
# Reads from st.secrets (Streamlit Cloud) or environment variables
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    CSV_OUTPUT_PATH   = st.secrets.get("CSV_OUTPUT_PATH", "earnings_analysis.csv")
except Exception:
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
    CSV_OUTPUT_PATH = "results/earnings_analysis.csv"




# Run data pipeline once (only if data file missing)
if "data_initialized" not in st.session_state:
    if not os.path.exists(CSV_OUTPUT_PATH):
        initialize_data()
    st.session_state["data_initialized"] = True


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Tone Intelligence · ORIK.AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e2e8f0; }
[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #1e2230; }

/* ── Login page ── */
.login-wrap {
    max-width: 420px; margin: 80px auto 0 auto;
    padding: 48px 40px; background: #111318;
    border: 1px solid #1e2230; border-radius: 6px;
}
.login-logo {
    font-family: 'IBM Plex Mono', monospace; font-size: 28px;
    color: #4a9eff; letter-spacing: -0.02em; margin-bottom: 4px;
}
.login-sub {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #4a5568; letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 36px;
}
.login-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #64748b; letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 4px;
}
.login-divider {
    border: none; border-top: 1px solid #1e2230; margin: 28px 0;
}

/* ── Branded header ── */
.app-header {
    display: flex; align-items: baseline; gap: 14px;
    padding-bottom: 16px; border-bottom: 1px solid #1e2230;
    margin-bottom: 24px;
}
.app-header-logo {
    font-family: 'IBM Plex Mono', monospace; font-size: 22px;
    color: #4a9eff; letter-spacing: -0.02em;
}
.app-header-title {
    font-family: 'IBM Plex Sans', sans-serif; font-weight: 300;
    font-size: 20px; color: #e2e8f0; letter-spacing: -0.01em;
}
.app-header-meta {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #4a5568; letter-spacing: 0.08em; margin-left: auto;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #111318; border: 1px solid #1e2230;
    border-radius: 4px; padding: 16px !important;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px !important;
    letter-spacing: 0.12em; color: #64748b !important; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 28px !important; color: #e2e8f0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: #111318 !important; border: 1px solid #1e2230;
    border-radius: 4px; margin-bottom: 8px;
}
[data-testid="stChatInput"] {
    background: #111318 !important; border: 1px solid #2d3748 !important;
    border-radius: 4px !important; font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ── Buttons ── */
.stButton button {
    background: #111318; border: 1px solid #2d3748; color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.05em;
    border-radius: 4px; padding: 8px 12px; width: 100%; text-align: left; transition: all 0.15s ease;
}
.stButton button:hover { border-color: #4a9eff; color: #4a9eff; background: #0d1929; }

/* ── Labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.15em;
    color: #4a9eff; text-transform: uppercase; margin-bottom: 12px;
    padding-bottom: 6px; border-bottom: 1px solid #1e2230;
}

/* ── Badges / tags ── */
.flag-positive {
    display: inline-block; background: #0a2a1a; border: 1px solid #10b981; color: #10b981;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.flag-negative {
    display: inline-block; background: #2a0a0a; border: 1px solid #ef4444; color: #ef4444;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.flag-neutral {
    display: inline-block; background: #151a27; border: 1px solid #4a5568; color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.phrase-tag {
    display: inline-block; background: #0d1929; border: 1px solid #1e3a5f; color: #7ab8ff;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-style: italic;
    padding: 3px 10px; border-radius: 2px; margin: 3px;
}
.industry-badge {
    display: inline-block; background: #1a1327; border: 1px solid #4c1d95; color: #a78bfa;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 10px; border-radius: 2px; margin: 2px;
}
.theme-tag {
    display: inline-block; background: #0d1a14; border: 1px solid #1a4731; color: #6ee7b7;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.hedge-high {
    display: inline-block; background: #2a1a06; border: 1px solid #d97706; color: #fbbf24;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.hedge-low {
    display: inline-block; background: #0a1a10; border: 1px solid #166534; color: #4ade80;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.hedge-term {
    display: inline-block; background: #1c1209; border: 1px solid #92400e; color: #f59e0b;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-style: italic;
    padding: 2px 7px; border-radius: 2px; margin: 2px;
}
.user-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0d1929; border: 1px solid #1e3a5f;
    border-radius: 20px; padding: 4px 12px 4px 8px;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #7ab8ff;
}
.user-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4a9eff; display: inline-block;
}
[data-testid="stExpander"] { background: #111318; border: 1px solid #1e2230; border-radius: 4px; }
hr { border-color: #1e2230 !important; }
h1 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 300; color: #e2e8f0; letter-spacing: -0.02em; }
h2 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; color: #cbd5e1; }
h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; color: #94a3b8; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1e2230; gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a5568; padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4a9eff; border-bottom: 2px solid #4a9eff;
}
</style>
""", unsafe_allow_html=True)

CHART_THEME = {
    "paper_bgcolor": "#0d0f14",
    "plot_bgcolor":  "#0d0f14",
    "font":          {"family": "IBM Plex Mono", "color": "#94a3b8", "size": 11},
    "gridcolor":     "#1e2230",
}

TICKER_COLORS = [
    "#4a9eff", "#10b981", "#f59e0b", "#a78bfa", "#f472b6",
    "#34d399", "#fb923c", "#60a5fa",
]

# ─────────────────────────────────────────────────────────────────────────────
# Login page
# ─────────────────────────────────────────────────────────────────────────────

def save_to_google_sheets(name: str, email: str) -> bool:
    try:
        # Get credentials from secrets
        creds_dict = st.secrets["google_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
        
        # Open the sheet
        sheet_id = st.secrets["google_sheet_id"]
        sheet = client.open_by_key(sheet_id).sheet1  # Assuming first sheet
        
        # Append row
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([name, email, timestamp])
        return True
    except Exception as e:
        st.error(f"Failed to save signup: {e}")
        return False


def show_login():
    # Hide sidebar on login page
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-wrap">
        <div class="login-logo">◈ ORIK.AI</div>
        <div class="login-sub">Earnings Tone Intelligence</div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown('<div class="login-label">Full Name</div>', unsafe_allow_html=True)
        name = st.text_input(
            "Name", placeholder="e.g. Sarah Chen",
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="login-label">Work Email</div>', unsafe_allow_html=True)
        email = st.text_input(
            "Email", placeholder="e.g. sarah@fundname.com",
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Enter →", use_container_width=True
        )

        if submitted:
            name  = name.strip()
            email = email.strip()
            if not name:
                st.error("Please enter your name.")
            elif not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                # Save to Google Sheets
                if save_to_google_sheets(name, email):
                    st.session_state["user_name"]  = name
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"]  = True
                    st.rerun()
                # If save fails, error is shown in the function

    st.markdown("""
        <hr class="login-divider">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2d3748;text-align:center;letter-spacing:0.1em;">
            CONFIDENTIAL · ORIK.AI · PERSONAL USE ONLY
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Gate: show login if not authenticated
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.get("logged_in"):
    show_login()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Hedging vocabulary
# ─────────────────────────────────────────────────────────────────────────────

HEDGE_GROUPS = {
    "Uncertainty":   [
        "uncertain", "uncertainty", "unclear", "unknown", "unpredictable",
        "hard to say", "difficult to predict", "hard to predict",
    ],
    "Conditionality": [
        "if", "assuming", "provided that", "subject to", "contingent",
        "depends on", "depending on", "conditional",
    ],
    "Approximation": [
        "approximately", "roughly", "around", "about", "estimated",
        "ballpark", "in the range of", "on the order of",
    ],
    "Weak modals":   [
        "may", "might", "could", "should", "possibly", "potentially",
        "perhaps", "we hope", "we expect", "we anticipate",
    ],
    "Downplay":      [
        "somewhat", "slightly", "modest", "moderate", "limited",
        "relatively", "to some extent", "to a degree", "in part",
    ],
    "Deflection":    [
        "we'll see", "wait and see", "monitor", "watching closely",
        "too early to say", "too early to tell", "remain cautious",
        "we're cautious", "prudent", "disciplined approach",
    ],
}

_ALL_HEDGE_TERMS = [t for terms in HEDGE_GROUPS.values() for t in terms]
_HEDGE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _ALL_HEDGE_TERMS) + r")\b",
    re.IGNORECASE,
)
_TERM_TO_GROUP: dict[str, str] = {
    t: g for g, terms in HEDGE_GROUPS.items() for t in terms
}


def detect_hedging(text: str) -> dict:
    words  = len(text.split()) or 1
    hits   = _HEDGE_RE.findall(text)
    unique = list(dict.fromkeys(h.lower() for h in hits))
    groups: Counter = Counter(_TERM_TO_GROUP.get(h.lower(), "Other") for h in hits)
    return {
        "count":   len(hits),
        "terms":   unique,
        "groups":  groups,
        "density": round(len(hits) / words * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCORE_COLS = [
    "management_overall_confidence",
    "management_prepared_remarks_sentiment",
    "management_responses_sentiment",
    "forward_guidance_sentiment",
    "analyst_pushback_level",
]

DELTA_COLS = [
    "delta_management_overall_confidence",
    "delta_management_prepared_remarks_sentiment",
    "delta_management_responses_sentiment",
    "delta_forward_guidance",
    "delta_analyst_pushback",
]

SCORE_LABELS = {
    "management_overall_confidence":         "Mgmt Confidence",
    "management_prepared_remarks_sentiment":  "Prepared Remarks",
    "management_responses_sentiment":         "Mgmt Responses",
    "forward_guidance_sentiment":             "Forward Guidance",
    "analyst_pushback_level":                 "Analyst Pushback",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ticker", "date"])
    for col in SCORE_COLS + DELTA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "industry" not in df.columns:
        df["industry"] = "Unknown"
    return df


if not os.path.exists(CSV_OUTPUT_PATH):
    st.error(f"No analysis CSV found at `{CSV_OUTPUT_PATH}`. Run `main.py` first.")
    st.stop()

df = load_csv(CSV_OUTPUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # User pill
    user_name  = st.session_state.get("user_name", "")
    user_email = st.session_state.get("user_email", "")
    st.markdown(
        f'<div class="user-pill"><span class="user-dot"></span>{user_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Industry</div>', unsafe_allow_html=True)
    all_industries = sorted(df["industry"].dropna().unique().tolist())
    selected_industries = st.multiselect(
        "Industries", options=all_industries, default=all_industries,
        label_visibility="collapsed",
    )
    show_industry_benchmark = st.toggle(
        "Show industry benchmark", value=True,
        help="Plots the median across all tracked companies in the same industry as a dashed line.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Companies</div>', unsafe_allow_html=True)
    industry_scoped_df = df[df["industry"].isin(selected_industries)] if selected_industries else df.copy()
    all_tickers = sorted(industry_scoped_df["ticker"].dropna().unique().tolist())
    selected_tickers = st.multiselect(
        "Tickers", options=all_tickers, default=all_tickers,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Date Range</div>', unsafe_allow_html=True)
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Display</div>', unsafe_allow_html=True)
    show_flags    = st.toggle("Change flags",        value=True)
    show_phrases  = st.toggle("Notable phrases",     value=True)
    show_themes   = st.toggle("Key themes",          value=True)
    show_hedging  = st.toggle("Hedging language",    value=True)
    show_raw      = st.toggle("Raw data table",      value=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sign out
    if st.button("⎋  Sign out"):
        for key in ["logged_in", "user_name", "user_email", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────

filtered_df = df[
    df["ticker"].isin(selected_tickers) &
    df["industry"].isin(selected_industries)
].copy() if (selected_tickers and selected_industries) else df.copy()

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["date"] >= pd.Timestamp(date_range[0])) &
        (filtered_df["date"] <= pd.Timestamp(date_range[1]))
    ]
filtered_df = filtered_df.sort_values("date")


def compute_industry_benchmarks(full_df: pd.DataFrame, dr) -> pd.DataFrame:
    bdf = full_df.copy()
    if len(dr) == 2:
        bdf = bdf[
            (bdf["date"] >= pd.Timestamp(dr[0])) &
            (bdf["date"] <= pd.Timestamp(dr[1]))
        ]
    bdf["period"] = bdf["year"].astype(str) + " Q" + bdf["quarter"].astype(str)
    return (
        bdf.groupby(["industry", "period", "year", "quarter"])[SCORE_COLS]
        .median().reset_index()
        .sort_values(["industry", "year", "quarter"])
    )


benchmarks = compute_industry_benchmarks(df, date_range)

# ─────────────────────────────────────────────────────────────────────────────
# Branded header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="app-header">'
    f'<span class="app-header-logo">◈</span>'
    f'<span class="app-header-title">Earnings Tone Intelligence</span>'
    f'<span class="app-header-meta">AI-scored sentiment · Confidence · Guidance · Pushback · Industry benchmarks</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tab structure
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_trends, tab_themes_hedging, tab_chat = st.tabs([
    "Overview",
    "Trends",
    "Themes & Hedging",
    "Ask the Data",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Per-ticker scorecard ───────────────────────────────────────────────
    if not filtered_df.empty and selected_tickers:
        st.markdown('<div class="section-label">Latest Quarter — Scorecard</div>', unsafe_allow_html=True)

        # Cap columns at 4; overflow to a second row
        chunk_size = 4
        ticker_chunks = [selected_tickers[i:i+chunk_size] for i in range(0, len(selected_tickers), chunk_size)]

        for chunk in ticker_chunks:
            ticker_cols = st.columns(len(chunk))
            for i, ticker in enumerate(chunk):
                tdf = filtered_df[filtered_df["ticker"] == ticker].sort_values("date")
                if tdf.empty:
                    continue
                last     = tdf.iloc[-1]
                industry = last.get("industry", "—")
                period   = f"{last['year']} Q{last['quarter']}"
                ind_bench = benchmarks[
                    (benchmarks["industry"] == industry) & (benchmarks["period"] == period)
                ]

                with ticker_cols[i]:
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                        f'letter-spacing:0.1em;color:#4a9eff;margin-bottom:4px;">'
                        f'{ticker}  ·  {period}</div>'
                        f'<span class="industry-badge">{industry}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    def fmt_delta(v):
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return None
                        return f"{v:+.0f} QoQ"

                    st.metric("Confidence",       f"{last['management_overall_confidence']:.0f}",
                              fmt_delta(last.get("delta_management_overall_confidence")))
                    st.metric("Guidance",         f"{last['forward_guidance_sentiment']:.0f}",
                              fmt_delta(last.get("delta_forward_guidance")))
                    st.metric("Analyst Pushback", f"{last['analyst_pushback_level']:.0f}",
                              fmt_delta(last.get("delta_analyst_pushback")))

                    if show_industry_benchmark and not ind_bench.empty:
                        med_conf  = ind_bench["management_overall_confidence"].iloc[0]
                        med_guide = ind_bench["forward_guidance_sentiment"].iloc[0]
                        conf_vs   = last["management_overall_confidence"] - med_conf
                        guide_vs  = last["forward_guidance_sentiment"] - med_guide
                        st.markdown(
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                            f'color:#4a5568;margin-top:8px;line-height:1.8;">'
                            f'Industry median<br>'
                            f'Conf: {med_conf:.0f} (<span style="color:{"#10b981" if conf_vs >= 0 else "#ef4444"}">{conf_vs:+.0f}</span>)<br>'
                            f'Guidance: {med_guide:.0f} (<span style="color:{"#10b981" if guide_vs >= 0 else "#ef4444"}">{guide_vs:+.0f}</span>)'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            st.markdown("<br>", unsafe_allow_html=True)

    elif filtered_df.empty:
        st.info("No data matches the current filters. Adjust the sidebar selections to see results.")

    # ── Change flags ───────────────────────────────────────────────────────
    if show_flags and not filtered_df.empty:
        flagged = filtered_df[
            filtered_df["change_flags"].notna() &
            ~filtered_df["change_flags"].str.contains("no material|first quarter", case=False, na=True)
        ].sort_values("date", ascending=False).head(8)

        if not flagged.empty:
            st.markdown('<div class="section-label">Notable QoQ Changes</div>', unsafe_allow_html=True)
            for _, row in flagged.iterrows():
                flags_list = [f.strip() for f in str(row.get("change_flags", "")).split(";") if f.strip()]
                badge_html = ""
                for flag in flags_list:
                    if "increased" in flag.lower():
                        badge_html += f'<span class="flag-positive">↑ {flag}</span>'
                    elif "decreased" in flag.lower():
                        badge_html += f'<span class="flag-negative">↓ {flag}</span>'
                    else:
                        badge_html += f'<span class="flag-neutral">{flag}</span>'
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#64748b;margin-right:8px;">'
                    f'{row["ticker"]}  {row["year"]} Q{row["quarter"]}</span>'
                    f'<span class="industry-badge">{row.get("industry","")}</span>'
                    f'<span style="margin-left:8px;">{badge_html}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Notable phrases ────────────────────────────────────────────────────
    if show_phrases and not filtered_df.empty:
        recent = filtered_df.sort_values("date", ascending=False).head(4)
        if recent["notable_phrases"].notna().any():
            st.markdown('<div class="section-label">Notable Phrases — Recent Quarters</div>', unsafe_allow_html=True)
            for _, row in recent.iterrows():
                phrases_raw = str(row.get("notable_phrases", ""))
                if not phrases_raw or phrases_raw == "nan":
                    continue
                phrases     = [p.strip() for p in phrases_raw.split(";") if p.strip()]
                phrase_html = "".join(f'<span class="phrase-tag">"{p}"</span>' for p in phrases)
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#4a5568;margin-right:10px;">'
                    f'{row["ticker"]} {row["year"]} Q{row["quarter"]}</span>'
                    f'{phrase_html}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Industry benchmark table ───────────────────────────────────────────
    if show_industry_benchmark and not filtered_df.empty:
        industries_in_view = filtered_df["industry"].dropna().unique()
        bench_view = benchmarks[benchmarks["industry"].isin(industries_in_view)].copy()
        bench_view["period"] = bench_view["year"].astype(str) + " Q" + bench_view["quarter"].astype(str)
        if not bench_view.empty:
            st.markdown('<div class="section-label">Industry Benchmarks — Period Medians</div>', unsafe_allow_html=True)
            display_bench = bench_view[["industry", "period"] + SCORE_COLS].sort_values(
                ["industry", "period"], ascending=[True, False]
            )
            display_bench.columns = ["Industry", "Period"] + [SCORE_LABELS.get(c, c) for c in SCORE_COLS]
            with st.expander("View industry benchmark table", expanded=False):
                st.dataframe(display_bench, use_container_width=True, hide_index=True)

    # ── Raw data ───────────────────────────────────────────────────────────
    if show_raw and not filtered_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Raw data table", expanded=False):
            display_cols = [
                "ticker", "industry", "date", "year", "quarter",
                "management_overall_confidence",          "delta_management_overall_confidence",
                "management_prepared_remarks_sentiment",   "delta_management_prepared_remarks_sentiment",
                "management_responses_sentiment",          "delta_management_responses_sentiment",
                "forward_guidance_sentiment",              "delta_forward_guidance",
                "analyst_pushback_level",                  "delta_analyst_pushback",
                "change_flags", "key_themes", "notable_phrases", "summary",
            ]
            existing_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[existing_cols].sort_values("date", ascending=False), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trends
# ═════════════════════════════════════════════════════════════════════════════

color_map = {t: TICKER_COLORS[i % len(TICKER_COLORS)] for i, t in enumerate(sorted(df["ticker"].unique()))}


def make_line_chart(company_df: pd.DataFrame, y_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for ticker in company_df["ticker"].unique():
        tdf   = company_df[company_df["ticker"] == ticker].sort_values("date")
        color = color_map.get(ticker, "#4a9eff")
        if tdf[y_col].isna().all():
            continue
        fig.add_trace(go.Scatter(
            x=tdf["period"], y=tdf[y_col],
            mode="lines+markers", name=ticker,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color, line=dict(color="#0d0f14", width=1)),
            hovertemplate=f"<b>{ticker}</b><br>%{{x}}<br>{title}: %{{y:.0f}}<extra></extra>",
        ))
    if show_industry_benchmark and y_col in benchmarks.columns:
        for industry in company_df["industry"].dropna().unique():
            ind_data = benchmarks[benchmarks["industry"] == industry].sort_values(["year", "quarter"])
            if ind_data.empty or ind_data[y_col].isna().all():
                continue
            fig.add_trace(go.Scatter(
                x=ind_data["period"], y=ind_data[y_col],
                mode="lines", name=f"{industry} median",
                line=dict(color="#4a5568", width=1, dash="dash"),
                hovertemplate=f"<b>{industry} median</b><br>%{{x}}<br>{title}: %{{y:.0f}}<extra></extra>",
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, family="IBM Plex Mono"), x=0),
        paper_bgcolor=CHART_THEME["paper_bgcolor"], plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        yaxis=dict(range=[0, 100], gridcolor=CHART_THEME["gridcolor"], tickfont=dict(size=10)),
        xaxis=dict(gridcolor=CHART_THEME["gridcolor"], tickangle=-30, tickfont=dict(size=10)),
        legend=dict(bgcolor="#111318", bordercolor="#1e2230", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=10)),
        margin=dict(l=0, r=0, t=36, b=0), height=260,
    )
    return fig


def make_delta_bar_chart(company_df: pd.DataFrame, delta_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for ticker in company_df["ticker"].unique():
        tdf = company_df[company_df["ticker"] == ticker].sort_values("date")
        tdf = tdf[tdf[delta_col].notna()]
        if tdf.empty:
            continue
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in tdf[delta_col]]
        fig.add_trace(go.Bar(
            x=tdf["period"], y=tdf[delta_col], name=ticker,
            marker_color=colors, opacity=0.85,
            hovertemplate=f"<b>{ticker}</b><br>%{{x}}<br>Δ: %{{y:+.1f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, family="IBM Plex Mono"), x=0),
        paper_bgcolor=CHART_THEME["paper_bgcolor"], plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        yaxis=dict(gridcolor=CHART_THEME["gridcolor"], tickfont=dict(size=10),
                   zeroline=True, zerolinecolor="#2d3748", zerolinewidth=1),
        xaxis=dict(gridcolor=CHART_THEME["gridcolor"], tickangle=-30, tickfont=dict(size=10)),
        legend=dict(bgcolor="#111318", bordercolor="#1e2230", borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=10)),
        barmode="group", margin=dict(l=0, r=0, t=36, b=0), height=260,
    )
    return fig


with tab_trends:
    chart_df = filtered_df.copy()
    chart_df["period"] = chart_df["year"].astype(str) + " Q" + chart_df["quarter"].astype(str)

    if chart_df.empty:
        st.info("No data matches the current filters.")
    else:
        st.markdown('<div class="section-label">Sentiment Trends</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_line_chart(chart_df, "management_overall_confidence",         "Management Confidence"), use_container_width=True)
        with c2:
            st.plotly_chart(make_line_chart(chart_df, "forward_guidance_sentiment",            "Forward Guidance Sentiment"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(make_line_chart(chart_df, "management_prepared_remarks_sentiment", "Prepared Remarks Sentiment"), use_container_width=True)
        with c4:
            st.plotly_chart(make_line_chart(chart_df, "analyst_pushback_level",                "Analyst Pushback Level"), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Quarter-over-Quarter Δ</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(make_delta_bar_chart(chart_df, "delta_management_overall_confidence", "Confidence Δ QoQ"), use_container_width=True)
        with c6:
            st.plotly_chart(make_delta_bar_chart(chart_df, "delta_forward_guidance",              "Guidance Δ QoQ"), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Themes & Hedging
# ═════════════════════════════════════════════════════════════════════════════

def extract_themes(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_in.iterrows():
        raw = str(row.get("key_themes", ""))
        if not raw or raw == "nan":
            continue
        period = f"{row['year']} Q{row['quarter']}"
        for theme in raw.split(";"):
            theme = theme.strip().lower()
            if not theme:
                continue
            rows.append({
                "ticker":  row["ticker"],
                "period":  period,
                "theme":   theme,
                "year":    row["year"],
                "quarter": row["quarter"],
            })
    return pd.DataFrame(rows)


with tab_themes_hedging:

    if filtered_df.empty:
        st.info("No data matches the current filters.")
    else:

        # ── Key themes ─────────────────────────────────────────────────────
        if show_themes and "key_themes" in filtered_df.columns:
            st.markdown('<div class="section-label">Key Themes — Management Focus</div>', unsafe_allow_html=True)
            theme_df = extract_themes(filtered_df)

            if not theme_df.empty:
                all_periods = sorted(
                    filtered_df[["year", "quarter"]].drop_duplicates().apply(
                        lambda r: f"{int(r['year'])} Q{int(r['quarter'])}", axis=1
                    ).tolist()
                )
                top_themes = [t for t, _ in Counter(theme_df["theme"]).most_common(10)]
                matrix = pd.DataFrame(0, index=top_themes, columns=all_periods)
                for _, row in theme_df.iterrows():
                    if row["theme"] in matrix.index and row["period"] in matrix.columns:
                        matrix.loc[row["theme"], row["period"]] += 1

                fig_heat = go.Figure(go.Heatmap(
                    z=matrix.values,
                    x=matrix.columns.tolist(),
                    y=matrix.index.tolist(),
                    colorscale=[
                        [0.0,  "#111318"],
                        [0.01, "#0d2a1a"],
                        [0.35, "#065f46"],
                        [0.7,  "#059669"],
                        [1.0,  "#6ee7b7"],
                    ],
                    showscale=False,
                    hovertemplate="<b>%{y}</b><br>%{x}<br>Mentions: %{z}<extra></extra>",
                ))
                fig_heat.update_layout(
                    paper_bgcolor=CHART_THEME["paper_bgcolor"],
                    plot_bgcolor=CHART_THEME["plot_bgcolor"],
                    font=CHART_THEME["font"],
                    xaxis=dict(
                        tickangle=-45, tickfont=dict(size=9),
                        gridcolor=CHART_THEME["gridcolor"],
                        tickmode="array", tickvals=all_periods, ticktext=all_periods,
                    ),
                    yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
                    margin=dict(l=0, r=0, t=8, b=0),
                    height=max(200, len(top_themes) * 30),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # Per-ticker dominant themes
                st.markdown(
                    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                    'color:#4a5568;letter-spacing:0.08em;margin-bottom:8px;">TOP THEMES BY COMPANY</div>',
                    unsafe_allow_html=True,
                )
                ticker_theme_cols = st.columns(max(1, len(selected_tickers)))
                for i, ticker in enumerate(selected_tickers):
                    t_themes = theme_df[theme_df["ticker"] == ticker]
                    if t_themes.empty:
                        continue
                    top = [t for t, _ in Counter(t_themes["theme"]).most_common(6)]
                    with ticker_theme_cols[i]:
                        st.markdown(
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                            f'color:#4a9eff;margin-bottom:6px;">{ticker}</div>'
                            + "".join(f'<span class="theme-tag">{t}</span>' for t in top),
                            unsafe_allow_html=True,
                        )

                # Theme shifts QoQ
                if len(all_periods) >= 2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                        'color:#4a5568;letter-spacing:0.08em;margin-bottom:10px;">'
                        'THEME SHIFTS — QUARTER-ON-QUARTER</div>',
                        unsafe_allow_html=True,
                    )
                    any_shifts = False
                    for idx in range(len(all_periods) - 1, 0, -1):
                        cur_period  = all_periods[idx]
                        prev_period = all_periods[idx - 1]
                        cur_themes  = set(theme_df[theme_df["period"] == cur_period]["theme"])
                        prev_themes = set(theme_df[theme_df["period"] == prev_period]["theme"])
                        new_this_q     = sorted(cur_themes  - prev_themes)
                        dropped_this_q = sorted(prev_themes - cur_themes)
                        if not new_this_q and not dropped_this_q:
                            continue
                        any_shifts = True
                        new_html = "".join(f'<span class="flag-positive">+ {t}</span>' for t in new_this_q)
                        dropped_html = "".join(f'<span class="flag-negative">− {t}</span>' for t in dropped_this_q)
                        st.markdown(
                            f'<div style="margin-bottom:10px;padding:8px 10px;'
                            f'background:#0d0f14;border:1px solid #1e2230;border-radius:4px;">'
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                            f'color:#4a9eff;margin-bottom:6px;">'
                            f'{cur_period} <span style="color:#4a5568;">vs</span> {prev_period}</div>'
                            f'{new_html}{dropped_html}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if not any_shifts:
                        st.markdown(
                            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                            'color:#4a5568;">No quarter-on-quarter theme changes detected.</span>',
                            unsafe_allow_html=True,
                        )

            st.markdown("<br>", unsafe_allow_html=True)

        # ── Hedging language ───────────────────────────────────────────────
        if show_hedging:
            st.markdown('<div class="section-label">Hedging Language & Tone Qualifiers</div>', unsafe_allow_html=True)

            hedge_rows = []
            for _, row in filtered_df.iterrows():
                text_to_scan = " ".join(filter(None, [
                    str(row.get("notable_phrases", "") or ""),
                    str(row.get("summary", "")         or ""),
                ]))
                if not text_to_scan.strip() or text_to_scan.strip() == "nan":
                    continue
                result = detect_hedging(text_to_scan)
                if result["count"] == 0:
                    continue
                hedge_rows.append({
                    "ticker":   row["ticker"],
                    "year":     row["year"],
                    "quarter":  row["quarter"],
                    "period":   f"{row['year']} Q{row['quarter']}",
                    "industry": row.get("industry", ""),
                    "count":    result["count"],
                    "density":  result["density"],
                    "terms":    result["terms"],
                    "groups":   result["groups"],
                    "summary":  str(row.get("summary", "") or ""),
                })

            if hedge_rows:
                hedge_df = pd.DataFrame(hedge_rows).sort_values("density", ascending=False)

                fig_hedge = go.Figure()
                for ticker in filtered_df["ticker"].unique():
                    tdata = hedge_df[hedge_df["ticker"] == ticker].sort_values(["year", "quarter"])
                    if tdata.empty:
                        continue
                    bar_colors = [
                        "#ef4444" if d >= 3.0 else "#f59e0b" if d >= 1.5 else "#4a9eff"
                        for d in tdata["density"]
                    ]
                    fig_hedge.add_trace(go.Bar(
                        x=tdata["period"], y=tdata["density"], name=ticker,
                        marker_color=bar_colors, opacity=0.85,
                        hovertemplate=(
                            f"<b>{ticker}</b><br>%{{x}}<br>"
                            "Hedge density: %{y:.2f} per 100 words<extra></extra>"
                        ),
                    ))
                fig_hedge.add_hline(y=1.5, line_dash="dash", line_color="#4a5568", line_width=1,
                                    annotation_text="Elevated", annotation_font=dict(color="#4a5568", size=9))
                fig_hedge.add_hline(y=3.0, line_dash="dash", line_color="#7f1d1d", line_width=1,
                                    annotation_text="High", annotation_font=dict(color="#7f1d1d", size=9))
                fig_hedge.update_layout(
                    title=dict(text="Hedging Density (per 100 words)", font=dict(size=12, family="IBM Plex Mono"), x=0),
                    paper_bgcolor=CHART_THEME["paper_bgcolor"], plot_bgcolor=CHART_THEME["plot_bgcolor"],
                    font=CHART_THEME["font"],
                    yaxis=dict(gridcolor=CHART_THEME["gridcolor"], tickfont=dict(size=10),
                               zeroline=True, zerolinecolor="#2d3748"),
                    xaxis=dict(gridcolor=CHART_THEME["gridcolor"], tickangle=-30, tickfont=dict(size=10)),
                    legend=dict(bgcolor="#111318", bordercolor="#1e2230", borderwidth=1,
                                font=dict(family="IBM Plex Mono", size=10)),
                    barmode="group", margin=dict(l=0, r=0, t=36, b=0), height=240,
                )
                st.plotly_chart(fig_hedge, use_container_width=True)

                st.markdown(
                    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                    'color:#4a5568;letter-spacing:0.08em;margin-bottom:8px;">'
                    'FLAGGED QUARTERS — HEDGING TERMS DETECTED</div>',
                    unsafe_allow_html=True,
                )
                for _, hrow in hedge_df.head(6).iterrows():
                    density_cls   = "hedge-high" if hrow["density"] >= 1.5 else "hedge-low"
                    density_label = f"{hrow['density']:.2f}/100w"
                    group_html = " ".join(
                        f'<span class="flag-neutral">{g} ×{c}</span>'
                        for g, c in sorted(hrow["groups"].items(), key=lambda x: -x[1])
                    )
                    terms_html = " ".join(f'<span class="hedge-term">{t}</span>' for t in hrow["terms"][:6])
                    st.markdown(
                        f'<div style="margin-bottom:12px;padding:10px;background:#0d0f14;'
                        f'border:1px solid #1e2230;border-radius:4px;">'
                        f'<div style="margin-bottom:6px;">'
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                        f'color:#64748b;margin-right:8px;">{hrow["ticker"]}  {hrow["period"]}</span>'
                        f'<span class="industry-badge">{hrow["industry"]}</span>'
                        f'<span class="{density_cls}" style="margin-left:8px;">{density_label}</span>'
                        f'</div>'
                        f'<div style="margin-bottom:4px;">{group_html}</div>'
                        f'<div>{terms_html}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#4a5568;">'
                    'No hedging language detected in selected quarters.</span>',
                    unsafe_allow_html=True,
                )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Ask the Data (Chatbot)
# ═════════════════════════════════════════════════════════════════════════════

def build_data_context(company_df: pd.DataFrame, bench_df: pd.DataFrame) -> str:
    if company_df.empty:
        return "No data available for the selected filters."

    lines = ["EARNINGS CALL ANALYSIS DATA\n"]
    lines.append(f"Total quarters: {len(company_df)}")
    lines.append(f"Tickers: {', '.join(sorted(company_df['ticker'].unique()))}")
    lines.append(f"Industries: {', '.join(sorted(company_df['industry'].dropna().unique()))}\n")

    for ticker, group in company_df.groupby("ticker"):
        group    = group.sort_values("date")
        industry = group["industry"].iloc[0] if "industry" in group.columns else "—"
        lines.append(f"=== {ticker}  [{industry}]  ({len(group)} quarters) ===")

        for _, row in group.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "unknown"

            def fv(col):
                v = row.get(col, None)
                return f"{v:.0f}" if v is not None and pd.notna(v) else "—"

            def fd(col):
                v = row.get(col, None)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return "—"
                return f"{v:+.0f}"

            period    = f"{row['year']} Q{row['quarter']}"
            ind_row   = bench_df[(bench_df["industry"] == industry) & (bench_df["period"] == period)]
            ind_conf  = f"{ind_row['management_overall_confidence'].iloc[0]:.0f}" if not ind_row.empty else "—"
            ind_guide = f"{ind_row['forward_guidance_sentiment'].iloc[0]:.0f}"    if not ind_row.empty else "—"

            hedge_text = " ".join(filter(None, [
                str(row.get("notable_phrases", "") or ""),
                str(row.get("summary", "") or ""),
            ]))
            hedge = detect_hedging(hedge_text)
            hedge_str = (
                f"hedging: {hedge['count']} hits ({hedge['density']:.1f}/100w), "
                f"groups=[{', '.join(f'{g}×{c}' for g,c in hedge['groups'].most_common(3))}], "
                f"terms=[{', '.join(hedge['terms'][:4])}]"
                if hedge["count"] > 0 else "hedging: none detected"
            )

            lines.append(
                f"  {date_str} Q{row.get('quarter','?')} | "
                f"Confidence: {fv('management_overall_confidence')} (Δ{fd('delta_management_overall_confidence')}, industry median: {ind_conf}) | "
                f"Prepared: {fv('management_prepared_remarks_sentiment')} (Δ{fd('delta_management_prepared_remarks_sentiment')}) | "
                f"Responses: {fv('management_responses_sentiment')} (Δ{fd('delta_management_responses_sentiment')}) | "
                f"Guidance: {fv('forward_guidance_sentiment')} (Δ{fd('delta_forward_guidance')}, industry median: {ind_guide}) | "
                f"Pushback: {fv('analyst_pushback_level')} (Δ{fd('delta_analyst_pushback')}) | "
                f"Flags: {row.get('change_flags','—')} | "
                f"Themes: {row.get('key_themes','—')} | "
                f"Notable phrases: {row.get('notable_phrases','—')} | "
                f"{hedge_str} | "
                f"Summary: {row.get('summary','—')}"
            )

        ticker_theme_df = company_df[company_df["ticker"] == ticker]
        all_themes: list[str] = []
        for _, trow in ticker_theme_df.iterrows():
            raw = str(trow.get("key_themes", "") or "")
            if raw and raw != "nan":
                all_themes.extend([t.strip().lower() for t in raw.split(";") if t.strip()])
        if all_themes:
            top = [f"{t} (×{c})" for t, c in Counter(all_themes).most_common(5)]
            lines.append(f"  Top themes across all quarters: {', '.join(top)}")

        if len(group) >= 2:
            first = group.iloc[0]
            last  = group.iloc[-1]
            for metric, label in [
                ("management_overall_confidence", "confidence"),
                ("forward_guidance_sentiment",    "guidance"),
            ]:
                try:
                    delta     = float(last[metric]) - float(first[metric])
                    direction = "improved" if delta > 0 else ("declined" if delta < 0 else "unchanged")
                    lines.append(f"  Long-run trend: {label} {direction} by {abs(delta):.0f} pts.")
                except (TypeError, ValueError):
                    pass
        lines.append("")

    industries_in_view = company_df["industry"].dropna().unique()
    if not bench_df.empty:
        lines.append("=== INDUSTRY BENCHMARKS (median across all tracked companies) ===")
        for industry in sorted(industries_in_view):
            idf = bench_df[bench_df["industry"] == industry].sort_values(["year", "quarter"])
            if idf.empty:
                continue
            lines.append(f"  {industry}:")
            for _, row in idf.iterrows():
                lines.append(
                    f"    {row['period']} | "
                    f"Confidence: {row['management_overall_confidence']:.0f} | "
                    f"Guidance: {row['forward_guidance_sentiment']:.0f} | "
                    f"Prepared: {row['management_prepared_remarks_sentiment']:.0f} | "
                    f"Pushback: {row['analyst_pushback_level']:.0f}"
                )
        lines.append("")

    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a senior equity analyst assistant for a buyside investment team,
specialising in qualitative earnings call analysis. You are speaking with {st.session_state.get('user_name', 'the analyst')}.

You have access to AI-scored sentiment data including scores, QoQ deltas, change flags,
key themes, notable phrases, hedging language signals, and industry benchmark medians.

Guidelines:
- Ground every answer in specific scores, dates, and delta values from the data.
- QoQ deltas are the primary signal — a 15-point confidence drop matters more than the absolute level.
- When a company diverges from its industry median, flag it explicitly.
  Example: "AAPL confidence at 82 vs Technology median 61 — significant premium."
- KEY THEMES: Identify which themes management returned to repeatedly across quarters.
  Recurring themes signal strategic priorities; newly absent themes signal pivots or problems.
  If asked about themes, reference specific quarters and note whether themes are new, recurring, or dropped.
- HEDGING LANGUAGE: The data includes hedging density (hits per 100 words) and hedging groups
  (Uncertainty, Conditionality, Weak modals, Deflection, etc.).
  High hedging density (>1.5/100w) in combination with a low confidence score is a strong
  caution signal. Hedging that increased QoQ — especially in Deflection or Conditionality —
  suggests management is pulling back on prior commitments. Always note the specific terms detected.
- Notable phrases are exact management language — quote them when they support the point.
- When comparing tickers, be explicit about divergences and convergences.
- Use analyst language: concise, evidence-based, no filler.
- Simple questions: 2–4 sentences. Complex multi-quarter/ticker/industry: short paragraphs.
- If data is insufficient, say so directly.
- Answer within 1,024 tokens

Score reference (all 0–100):
  management_overall_confidence:            0=heavily hedged, 50=neutral, 100=highly assertive
  management_prepared_remarks_sentiment:    0=very negative, 50=neutral, 100=very positive
  management_responses_sentiment:           0=very negative, 50=neutral, 100=very positive
  forward_guidance_sentiment:               0=guiding down, 50=neutral, 100=guiding up
  analyst_pushback_level:                   0=softball, 50=neutral, 100=aggressive scrutiny
  delta_* fields:                           positive=improvement QoQ, negative=deterioration
  change_flags:                             fields that moved >10 pts QoQ — treat as alerts
  hedging density:                          hits per 100 words; >1.5 = elevated, >3.0 = high
  industry median:                          median score across all tracked peers in same sector"""


with tab_chat:
 
    # Chat header with clear button inline
    chat_header_col, clear_col = st.columns([6, 1])
    with chat_header_col:
        st.markdown('<div class="section-label">Ask the Data</div>', unsafe_allow_html=True)
    with clear_col:
        if st.button("↺ Clear", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
 
    if filtered_df.empty:
        st.warning("No data loaded — adjust the sidebar filters before asking questions.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
 
        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
 
        # Suggested queries (only shown before first message)
        if not st.session_state.messages:
            suggestions = []
 
            delta_col = "delta_management_overall_confidence"
            if delta_col in filtered_df.columns:
                valid = filtered_df[filtered_df[delta_col].notna()]
                if not valid.empty:
                    worst     = valid.loc[valid[delta_col].abs().idxmax()]
                    direction = "drop" if worst[delta_col] < 0 else "jump"
                    suggestions.append(
                        f"What drove the confidence {direction} for {worst['ticker']} in {worst['year']} Q{worst['quarter']}?"
                    )
 
            if "key_themes" in filtered_df.columns:
                all_t: list[str] = []
                for _, row in filtered_df.iterrows():
                    raw = str(row.get("key_themes", "") or "")
                    if raw and raw != "nan":
                        all_t.extend([t.strip().lower() for t in raw.split(";") if t.strip()])
                if all_t:
                    top_theme = Counter(all_t).most_common(1)[0][0]
                    suggestions.append(f"Which companies discussed '{top_theme}' most, and how did their tone differ?")
 
            hedge_scan = []
            for _, row in filtered_df.iterrows():
                ht = " ".join(filter(None, [str(row.get("notable_phrases","") or ""), str(row.get("summary","") or "")]))
                h  = detect_hedging(ht)
                if h["count"] > 0:
                    hedge_scan.append((row["ticker"], row["year"], row["quarter"], h["density"]))
            if hedge_scan:
                hedge_scan.sort(key=lambda x: -x[3])
                ht, hy, hq, _ = hedge_scan[0]
                suggestions.append(f"{ht} had the highest hedging density in {hy} Q{hq} — what were they cautious about?")
 
            if len(selected_tickers) >= 2:
                suggestions.append(
                    f"Compare guidance sentiment for {selected_tickers[0]} vs {selected_tickers[1]} "
                    f"against their industry benchmarks"
                )
 
            if len(suggestions) < 4:
                suggestions.append("Which companies show consistently above-industry confidence, and which are laggards?")
 
            st.markdown(
                '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                'color:#4a5568;letter-spacing:0.1em;">START WITH A SUGGESTED QUERY</span>',
                unsafe_allow_html=True,
            )
            sug_cols = st.columns(2)
            for i, s in enumerate(suggestions[:4]):
                # CHANGE 1: clicking a suggestion appends the user message and reruns —
                # the run_chat_completion() call below then fires automatically on rerun.
                if sug_cols[i % 2].button(s, key=f"sug_{i}"):
                    st.session_state.messages.append({"role": "user", "content": s})
                    st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
 
        # ── Shared helper: build context + stream a response ──────────────
        def run_chat_completion():
            """Rebuild data context and stream the assistant reply for the latest user message."""
            bench_with_period = benchmarks.copy()
            if "period" not in bench_with_period.columns:
                bench_with_period["period"] = (
                    bench_with_period["year"].astype(str) + " Q" + bench_with_period["quarter"].astype(str)
                )
            data_context = build_data_context(filtered_df, bench_with_period)
 
            # Inject context into first user message of every API call
            api_messages = []
            for i, msg in enumerate(st.session_state.messages):
                content = msg["content"]
                if i == 0 and msg["role"] == "user":
                    content = (
                        f"Here is the earnings call analysis data:\n\n{data_context}\n\n"
                        f"---\n\nQuestion: {content}"
                    )
                api_messages.append({"role": msg["role"], "content": content})
 
            with st.chat_message("assistant"):
                try:
                    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                    # CHANGE 2: stream the response token-by-token
                    with client.messages.stream(
                        model      = "claude-sonnet-4-6",
                        max_tokens = 1024,
                        system     = SYSTEM_PROMPT,
                        messages   = api_messages,
                    ) as stream:
                        reply = st.write_stream(stream.text_stream)
                except anthropic.AuthenticationError:
                    reply = "⚠ Authentication failed — check that your `ANTHROPIC_API_KEY` is set correctly in `st.secrets` or your environment."
                    st.markdown(reply)
                except anthropic.RateLimitError:
                    reply = "⚠ Rate limit reached. Wait a moment and try again."
                    st.markdown(reply)
                except Exception as e:
                    reply = f"⚠ Unexpected error: {e}"
                    st.markdown(reply)
 
            st.session_state.messages.append({"role": "assistant", "content": reply})
 
        # CHANGE 1 (continued): after a suggestion rerun, the last message is a user message
        # with no assistant reply yet — detect that and fire completion immediately.
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            run_chat_completion()
 
        # Normal chat input path
        if user_input := st.chat_input("Query earnings tone data ..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            run_chat_completion()
