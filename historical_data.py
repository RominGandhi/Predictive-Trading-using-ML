import yfinance as yf
import pandas as pd
import json
import numpy as np
import os

def download_and_save_historical_data(ticker, start_date, end_date):
    tickers = [t.strip().upper() for t in ticker.split(",")]

    # Create folder if it doesn't exist
    output_dir = os.path.join("data", "historical_data")
    os.makedirs(output_dir, exist_ok=True)

    for t in tickers:
        print(f"\nDownloading data for {t} from {start_date} to {end_date}...\n")
        data = yf.download(t, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for {t}. Skipping.")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Calculate return and volatility
        data['Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Return'].rolling(window=30).std()

        # Drop only if Close is missing
        data_cleaned = data.dropna(subset=["Close"]).copy()

        # Reset index and convert Date to string
        data_cleaned.reset_index(inplace=True)
        data_cleaned['Date'] = data_cleaned['Date'].astype(str)

        # Convert NaNs to None for JSON compatibility
        data_dict = data_cleaned.replace({np.nan: None}).to_dict(orient='records')

        # Save to JSON
        json_filename = os.path.join(output_dir, f"{t}_historical_data.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

        print(f"✅ Saved to {json_filename}")
