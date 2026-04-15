# config.py
# ------------------------------------------------------------
# Central config — API keys and shared paths.
# Never share or commit this file.
# ------------------------------------------------------------


import os
#from dotenv import load_dotenv

#load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CSV_OUTPUT_PATH = "results/earnings_analysis.csv"
