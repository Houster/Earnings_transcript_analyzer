# Earnings Sentiment Analyzer

A comprehensive tool for analyzing earnings call transcripts to extract sentiment, confidence levels, and forward guidance using AI-powered analysis and financial sentiment dictionaries.

## Features

- **Automated Transcript Fetching**: Pulls earnings call transcripts from multiple sources for specified tickers and time periods
- **AI-Powered Analysis**: Uses Anthropic's Claude AI to analyze management tone, confidence, and forward guidance
- **Sentiment Scoring**: Implements Loughran-McDonald financial sentiment dictionary for word-level sentiment analysis
- **Industry Classification**: Automatically categorizes companies into industry sectors
- **Change Detection**: Identifies quarter-over-quarter changes in sentiment and guidance
- **Interactive Chatbot**: Streamlit-based web interface for exploring analysis results
- **Data Export**: Outputs results in CSV format and individual JSON files for each quarter

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/earnings-sentiment-analyzer.git
cd earnings-sentiment-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Get an API key from [Anthropic Console](https://console.anthropic.com)
   - Create a `.env` file or set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   - For Streamlit deployment, add to `st.secrets`
   - MSFT and AAPL can be accessed for free. For more companies, an EarningsCall API is required

## Usage

### Running Analysis

Execute the main analysis pipeline:
```bash
python main.py
```

This will analyze the last year's earnings calls for Microsoft (MSFT) by default. Modify `main.py` to change tickers or time periods.


### Interactive Chatbot

Launch the Streamlit chatbot interface:
```bash
streamlit run chatbot.py
```

The chatbot provides:
- Company and industry filtering
- Sentiment trend visualization
- Detailed quarter-by-quarter analysis
- Confidence and guidance metrics

## Configuration

Key settings in `config.py`:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `CSV_OUTPUT_PATH`: Path for CSV output file

## Data Sources

- **Transcripts**: Fetched using the `earningscall` library
- **Sentiment Dictionary**: Loughran-McDonald Master Dictionary (included in `lm_dictionary/`)
- **AI Analysis**: Anthropic Claude for contextual understanding

## Project Structure

```
earnings-sentiment-analyzer/
├── analyzer.py          # Main analysis pipeline
├── chatbot.py           # Streamlit web interface
├── config.py            # Configuration and API keys
├── fetcher.py           # Transcript fetching logic
├── lm_scorer.py         # Loughran-McDonald sentiment scoring
├── main.py              # Entry point for analysis
├── transcript_parser.py # Transcript parsing utilities
├── requirements.txt     # Python dependencies
├── lm_dictionary/       # Financial sentiment dictionary
├── results/             # Output files (JSON and CSV)
└── README.md           # This file
```

## Output Format

Results are saved in two formats:

1. **CSV File** (`results/earnings_analysis.csv`): Tabular data for all analyzed quarters
2. **JSON Files**: Individual files per quarter with detailed analysis (e.g., `AAPL_2024_Q1_analysis.json`)

Each analysis includes:
- Management confidence scores
- Forward guidance sentiment
- Key themes and topics
- Sentiment breakdowns
- Industry classification

## Dependencies

- `anthropic`: Claude AI integration
- `requests`: HTTP requests for data fetching
- `pandas`: Data manipulation and CSV export
- `streamlit`: Web interface for chatbot
- `pymupdf`: PDF processing for transcripts
- `plotly`: Data visualization
- `earningscall`: Earnings transcript API

## Security Note

⚠️ **Important**: Never commit API keys to version control. The `config.py` file contains a placeholder API key for demonstration only. Use environment variables or Streamlit secrets for production deployments.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Contact

For questions or support, please open an issue on GitHub.