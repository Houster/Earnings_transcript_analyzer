# main.py
# ------------------------------------------------------------
# Run this file to pull and analyse earnings calls.
# Usage: python main.py
# Then launch the chatbot: streamlit run chatbot.py
# ------------------------------------------------------------

from analyzer import run_analysis_multi_year



def initialize_data():
    results = run_analysis_multi_year(ticker="MSFT", years=10)
    return results

