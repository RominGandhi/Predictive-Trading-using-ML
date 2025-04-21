import requests
import os
import json
import time
from dotenv import load_dotenv

# Load API key
load_dotenv("keys.env")
FMP_API_KEY = os.getenv("FMP_API_KEY")

BASE_URL = "https://financialmodelingprep.com/api/v3"
DATA_DIR = "data/statements"

def fetch_statement(ticker, statement_type, period="annual", limit=10):
    url = f"{BASE_URL}/{statement_type}/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        print(f"Retrieved {statement_type.replace('-', ' ').title()} for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching {statement_type} for {ticker}: {e}")
        return []

def save_to_file(data, ticker, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"{ticker}_{name}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {filename}")

def get_full_financials(ticker_input):
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    statements = {
        "income": "income-statement",
        "balance": "balance-sheet-statement",
        "cashflow": "cash-flow-statement"
    }

    for ticker in tickers:
        print(f"\n📊 Fetching financial statements for {ticker}")
        for key, endpoint in statements.items():
            data = fetch_statement(ticker, endpoint, period="annual", limit=5)
            if data:
                save_to_file(data, ticker, key)
            time.sleep(0.2)


# Optional: keep for testing standalone runs
if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    get_full_financials(ticker)
