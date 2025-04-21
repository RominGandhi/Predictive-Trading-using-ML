import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_price_df(ticker, historical_dir):
    path = os.path.join(historical_dir, f"{ticker}_historical_data.json")
    df = pd.DataFrame(load_json(path))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_technical_indicators(df, window=30):
    df["Return_1D"] = df["Close"].pct_change()
    df["RollingMean"] = df["Close"].rolling(window=window).mean()
    df["RollingStd"] = df["Close"].rolling(window=window).std()
    df["Volatility"] = df["RollingStd"] / df["RollingMean"]
    df["Momentum"] = df["Close"] / df["Close"].shift(window) - 1
    return df

def compute_multi_target_returns(df):
    for horizon in [30, 60, 90, 120]:
        df[f"Target_Return_{horizon}D"] = df["Close"].shift(-horizon) / df["Close"] - 1
    return df

def load_sentiment_df(ticker, sentiment_dir):
    path = os.path.join(sentiment_dir, f"{ticker}_news_sentiment.json")
    if not os.path.exists(path):
        print(f"⚠️ No sentiment file found for {ticker}")
        return pd.DataFrame(columns=["published_at", "sentiment_score"])

    data = load_json(path)
    df = pd.DataFrame(data)
    return df

def compute_sentiment_features_daily(sentiment_df, price_df):
    # Standardize both dates to datetime64[ns]
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()

    if sentiment_df.empty:
        print("⚠️ Sentiment dataframe is empty — filling with 0s.")
        price_df["Sentiment_Avg"] = 0
        price_df["Sentiment_Std"] = 0
        price_df["Sentiment_Count"] = 0
        return price_df

    # Extract & normalize sentiment date
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["published_at"]).dt.normalize()

    # Group by day and aggregate
    daily_sentiment = sentiment_df.groupby("Date").agg({
        "sentiment_score": ["mean", "std", "count"]
    })

    daily_sentiment.columns = ["Sentiment_Avg", "Sentiment_Std", "Sentiment_Count"]
    daily_sentiment = daily_sentiment.reset_index()

    # Merge
    merged = pd.merge(price_df, daily_sentiment, on="Date", how="left")

    # Fill any missing sentiment days
    merged["Sentiment_Avg"] = merged["Sentiment_Avg"].fillna(0)
    merged["Sentiment_Std"] = merged["Sentiment_Std"].fillna(0)
    merged["Sentiment_Count"] = merged["Sentiment_Count"].fillna(0)

    # Sanity check
    print(f"✅ Sentiment merged. Sample rows:\n{merged[['Date', 'Sentiment_Avg', 'Sentiment_Count']].tail(3)}")

    return merged

def extract_by_year(data, year: str):
    for entry in data:
        if entry.get("calendarYear") == year:
            return entry
    return None

def extract_financial_features(balance, income, cashflow):
    try:
        revenue = income.get("revenue", 0)
        net_income = income.get("netIncome", 0)
        op_income = income.get("operatingIncome", 0)
        equity = balance.get("totalStockholdersEquity", 0)
        liabilities = balance.get("totalLiabilities", 0)
        curr_assets = balance.get("totalCurrentAssets", 0)
        curr_liab = balance.get("totalCurrentLiabilities", 0)
        fcf = cashflow.get("freeCashFlow", 0)

        return {
            "ROE": net_income / equity if equity else 0,
            "Net_Margin": net_income / revenue if revenue else 0,
            "Op_Margin": op_income / revenue if revenue else 0,
            "Debt_to_Equity": liabilities / equity if equity else 0,
            "Current_Ratio": curr_assets / curr_liab if curr_liab else 0,
            "FCF_Margin": fcf / revenue if revenue else 0
        }
    except Exception as e:
        print(f"⚠️ Error computing financial ratios: {e}")
        return {
            "ROE": 0, "Net_Margin": 0, "Op_Margin": 0,
            "Debt_to_Equity": 0, "Current_Ratio": 0, "FCF_Margin": 0
        }
