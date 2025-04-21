import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_loader import *
from models.gradient_boosting_model import *
from models.lstm_model import *
import altair as alt
from data_fetch_FMP import get_full_financials
from historical_data import download_and_save_historical_data
from news_sentiment import run_news_sentiment_pipeline
from datetime import datetime

st.set_page_config(page_title="Predictive Trading Strategy", layout="wide")

def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# Initialize session state
if "data_fetched" not in st.session_state:
    st.session_state["data_fetched"] = False

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color: #FAFAFA;'>Configuration</h2>", unsafe_allow_html=True)

    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    start_date = st.text_input("Start Date (YYYY-MM-DD)", value="2020-01-01")
    default_end = datetime.today().strftime('%Y-%m-%d')
    end_date = st.text_input("End Date (YYYY-MM-DD)", value=default_end)

    user_news_input = st.text_input("Enter impactful news events (comma-separated)", "")

    seed = st.slider("Random Seed", 0, 100, 42)
    monte_carlo_runs = st.slider("Monte Carlo Simulations", 10, 1000, 100, step=10)

    st.markdown(
        "<p style='font-size: 12px; color: #888;'>This dashboard uses financials, sentiment, and price action to predict short-term stock returns using Gradient Boosting and LSTM models.</p>",
        unsafe_allow_html=True
    )

    # Spacer to push button to bottom
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0.5px solid #555;'>", unsafe_allow_html=True)

    # Custom fetch button container
    st.markdown("<div class='fetch-button-wrapper'>", unsafe_allow_html=True)

    if not st.session_state.get("data_fetched", False):
        if st.button("Fetch Data"):
            if ticker and start_date and end_date:
                with st.spinner(f"Fetching data for {ticker}..."):
                    try:
                        get_full_financials(ticker)
                        download_and_save_historical_data(ticker, start_date, end_date)
                        run_news_sentiment_pipeline(ticker)
                        st.success(f"Data successfully fetched for {ticker}")
                        st.session_state["data_fetched"] = True
                    except Exception as e:
                        st.sidebar.error(f"Failed to fetch data for {ticker}: {e}")
    else:
        st.success(f"Data already loaded for {ticker}")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Paths & config ---
BASE_PATH = "data"
PRICE_PATH = os.path.join(BASE_PATH, "historical_data")
SENTIMENT_PATH = os.path.join(BASE_PATH, "sentiment")
STATEMENT_PATH = os.path.join(BASE_PATH, "statements")

target_cols = ["Target_Return_30D", "Target_Return_60D", "Target_Return_90D", "Target_Return_120D"]
feature_cols = [
    "Return_1D", "RollingMean", "RollingStd", "Volatility", "Momentum",
    "Sentiment_Avg", "Sentiment_Std", "Sentiment_Count",
    "ROE", "Net_Margin", "Op_Margin", "Debt_to_Equity", "Current_Ratio", "FCF_Margin"
]

np.random.seed(seed)
torch.manual_seed(seed)

# --- Forecast Chart ---


def streamlit_forecast_chart(price_df, gb_preds, lstm_preds):
    # Prepare historical price data
    price_df = price_df.copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    last_close = price_df["Close"].iloc[-1]
    last_date = price_df["Date"].iloc[-1]

    # Filter last 5 years of historical prices
    cutoff_date = last_date - pd.DateOffset(years=5)
    price_df = price_df[price_df["Date"] >= cutoff_date]

    # Generate forecast prices
    horizons = [30, 60, 90, 120]
    future_dates = [last_date + pd.Timedelta(days=h) for h in horizons]
    gb_prices = [last_close * (1 + gb_preds[f"Target_Return_{h}D"]["mean"]) for h in horizons]
    lstm_prices = [last_close * (1 + lstm_preds[f"Target_Return_{h}D"]["mean"]) for h in horizons]

    # Forecast DataFrame
    df_forecast = pd.DataFrame({
        "Date": future_dates * 2,
        "Price": gb_prices + lstm_prices,
        "Model": ["GBM Forecast"] * 4 + ["LSTM Forecast"] * 4
    })

    # Historical DataFrame
    df_hist = price_df[["Date", "Close"]].rename(columns={"Close": "Price"})
    df_hist["Model"] = "Historical"

    # Merge and sort
    df_combined = pd.concat([df_hist, df_forecast])
    df_combined.sort_values("Date", inplace=True)

    # Display chart title with spacing
    st.markdown("<h4 style='color:white; font-size:28px; margin-top: 30px; margin-bottom: 40px;'>Forecasted Stock Price (Last 5 Years + Predictions)</h4>", unsafe_allow_html=True)

    # Build Altair black-themed line chart
    chart = alt.Chart(df_combined).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Price:Q", title="Stock Price"),
        color=alt.Color("Model:N", scale=alt.Scale(scheme='category10')),
        tooltip=["Date:T", "Price:Q", "Model:N"]
    ).properties(
        width=750,
        height=500,
        background="#000000"
    ).configure_view(
        strokeWidth=0,
        fill="#000000"
    ).configure_axis(
        grid=True,
        gridColor="#222222",
        domain=False,
        labelColor='white',
        titleColor='white'
    ).configure_legend(
        labelColor='white',
        titleColor='white',
        orient="top"
    ).configure_title(
        color='white',
        fontSize=18,
        anchor='start'
    )

    st.altair_chart(chart, use_container_width=True)


# --- Model Runner ---
@st.cache_data(show_spinner=False)
def run_model_pipeline(ticker, monte_carlo_runs):
    from utils.data_loader import extract_by_year, extract_financial_features
    from models.lstm_model import monte_carlo_lstm_predict

    price_df = load_price_df(ticker, PRICE_PATH)
    price_df = compute_sentiment_features_daily(load_sentiment_df(ticker, SENTIMENT_PATH), price_df)
    price_df = add_technical_indicators(price_df)
    price_df = compute_multi_target_returns(price_df)

    income = load_json(os.path.join(STATEMENT_PATH, f"{ticker}_income.json"))
    balance = load_json(os.path.join(STATEMENT_PATH, f"{ticker}_balance.json"))
    cashflow = load_json(os.path.join(STATEMENT_PATH, f"{ticker}_cashflow.json"))

    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df["Year"] = price_df["Date"].dt.year
    for col in ["ROE", "Net_Margin", "Op_Margin", "Debt_to_Equity", "Current_Ratio", "FCF_Margin"]:
        price_df[col] = np.nan

    for year in price_df["Year"].unique():
        i = extract_by_year(income, str(year))
        b = extract_by_year(balance, str(year))
        c = extract_by_year(cashflow, str(year))
        if i and b and c:
            feats = extract_financial_features(b, i, c)
            for col, val in feats.items():
                price_df.loc[price_df["Year"] == year, col] = val

    price_df.ffill(inplace=True)

    gb_models, gb_perf = train_multi_boosting_models(price_df, feature_cols, target_cols)
    latest_X = price_df[feature_cols].dropna().iloc[[-1]]
    gb_preds = {col: monte_carlo_predict(model, latest_X, iterations=monte_carlo_runs) for col, model in gb_models.items()}


    lstm_preds = {}
    lstm_perf = {}

    for col in target_cols:
        X_lstm, y_lstm = prepare_lstm_data(price_df, feature_cols, col)
        split = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split], X_lstm[split:]
        y_train, y_test = y_lstm[:split], y_lstm[split:]

        model = train_lstm_model(X_train, y_train, input_size=len(feature_cols))

        with torch.no_grad():
            y_pred = model(X_test).squeeze().numpy()

        # Monte Carlo prediction
        x_latest_seq = X_lstm[-1].unsqueeze(0)
        lstm_mc_result = monte_carlo_lstm_predict(model, x_latest_seq, iterations=monte_carlo_runs)

        rmse = mean_squared_error(y_test.numpy(), y_pred)
        r2 = r2_score(y_test.numpy(), y_pred)

        lstm_preds[col] = lstm_mc_result
        lstm_perf[col] = {"RMSE": rmse, "R2": r2}

    feature_importance = {
        col.replace("Target_Return_", ""): dict(zip(feature_cols, gb_models[col].feature_importances_))
        for col in target_cols
    }

    return gb_preds, lstm_preds, feature_importance, gb_perf, lstm_perf, price_df

# --- UI ---
st.markdown("""
    <h1 style='font-size: 38px; color: #FFDD57; margin-bottom: 10px;'>Predictive Trading Strategy </h1>
    <p style='font-size: 16px; color: #f0f0f0;'>
        This dashboard provides a data-driven forecasting environment that uses alternative data sources and machine learning models 
        to estimate short-term stock returns over the next <strong>30, 60, 90, and 120 days</strong>.
        <br><br>
        Leveraging financial fundamentals, market sentiment, and technical indicators, we compare predictions from two different models:
        <strong>Gradient Boosting</strong> and <strong>LSTM Neural Networks</strong>.
        Forecasts are generated using <strong>Monte Carlo simulations</strong> to reflect uncertainty and return distribution.
        <br><br>
        <em>This tool is designed for research, analysis, and educational exploration — not financial advice.</em>
    </p>
""", unsafe_allow_html=True)

if ticker:
    try:
        with st.spinner("Running models..."):
            gb_preds, lstm_preds, feat_importance, gb_perf, lstm_perf, price_df = run_model_pipeline(ticker, monte_carlo_runs)

        st.markdown(f"<h3 style='margin-top: 20px;'>Forecast Results for: <span style='color:#FFDD57;'>{ticker}</span></h3>", unsafe_allow_html=True)

        for horizon in ["30", "60", "90", "120"]:
            gb = gb_preds[f"Target_Return_{horizon}D"]
            lstm = lstm_preds[f"Target_Return_{horizon}D"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background-color:#111;padding:15px 25px;border:1px solid #444;border-radius:8px;margin-bottom:15px;'>
                    <div style='color:#ccc;font-size:15px;font-weight:bold;'>{horizon}D Gradient Boosting</div>
                    <div style='color:#FAFAFA;font-size:20px;'>{gb['mean']:+.2%} <span style='font-size:14px;color:#888'>(±{gb['std']:.2%})</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background-color:#111;padding:15px 25px;border:1px solid #444;border-radius:8px;margin-bottom:15px;'>
                    <div style='color:#ccc;font-size:15px;font-weight:bold;'>{horizon}D LSTM</div>
                    <div style='color:#FAFAFA;font-size:20px;'>{lstm["mean"]:+.2%} 
                    <span style='font-size:14px;color:#888'>(±{lstm["std"]:.2%})</span></div>

                </div>
                """, unsafe_allow_html=True)


        st.markdown(
            f"""
            <div class="monte-carlo-box">
                <h4 style='margin-bottom:10px;'>Monte Carlo Simulation</h4>
                <p style='font-size:15px;'>
                    Forecasts shown above for <strong>both Gradient Boosting and LSTM models</strong> are based on Monte Carlo simulation with <code>{monte_carlo_runs} runs</code>.
                </p>
                <p style='font-size:14px; color:#cccccc; margin-bottom:0;'>
                    The mean and standard deviation (±) displayed for each model are derived from the distribution of predictions over these runs to reflect uncertainty and variance in short-term stock return forecasts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


        # --- Line Chart for Forecasted Price ---
        streamlit_forecast_chart(price_df, gb_preds, lstm_preds)

        # --- Monte Carlo Summary ---

        # --- Recommendation ---
        avg_r2_gbm = np.mean([v["R2"] for v in gb_perf.values()])
        avg_r2_lstm = np.mean([v["R2"] for v in lstm_perf.values()])

        if avg_r2_gbm > avg_r2_lstm:
            better_model = "Gradient Boosting"
            r2_diff = avg_r2_gbm - avg_r2_lstm
        else:
            better_model = "LSTM"
            r2_diff = avg_r2_lstm - avg_r2_gbm

        st.markdown(
            f"""
            <div class='model-summary'>
                <h4> Model Accuracy Summary</h4>
                <p style='font-size:15px; margin-bottom:8px;'>
                    Based on the average <strong>R² scores</strong> across all forecast horizons, the <strong>{better_model}</strong> model demonstrated superior performance.
                </p>
                <p>
                    The difference in average R² between the two models is <strong>{r2_diff:.4f}</strong>. This indicates that <strong>{better_model}</strong> more accurately explains the variance in stock returns for <code>{ticker}</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Performance Section ---
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        st.markdown("### Comprehensive Model Performance")

        def perf_color(value, better, mode="r2"):
            if mode == "r2":
                return "#10c466" if value > better else "#ff4b5c"
            else:
                return "#10c466" if value < better else "#ff4b5c"

        for h in ["30", "60", "90", "120"]:
            gb = gb_perf[f"Target_Return_{h}D"]
            lstm = lstm_perf[f"Target_Return_{h}D"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background-color:#111;padding:20px;border:1px solid #444;border-radius:8px;margin-bottom:15px;'>
                    <div style='color:#ccc;font-weight:bold;font-size:16px;'>GBM - {h}D Horizon</div>
                    <div style='margin-top:10px;font-size:14px;'>
                        <span style='color:#888;'>RMSE:</span>
                        <span style='color:{perf_color(gb['RMSE'], lstm['RMSE'], "rmse")}; font-weight:bold;'> {gb['RMSE']:.4f}</span><br>
                        <span style='color:#888;'>R²:</span>
                        <span style='color:{perf_color(gb['R2'], lstm['R2'], "r2")}; font-weight:bold;'> {gb['R2']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background-color:#111;padding:20px;border:1px solid #444;border-radius:8px;margin-bottom:15px;'>
                    <div style='color:#ccc;font-weight:bold;font-size:16px;'>LSTM - {h}D Horizon</div>
                    <div style='margin-top:10px;font-size:14px;'>
                        <span style='color:#888;'>RMSE:</span>
                        <span style='color:{perf_color(lstm['RMSE'], gb['RMSE'], "rmse")}; font-weight:bold;'> {lstm['RMSE']:.4f}</span><br>
                        <span style='color:#888;'>R²:</span>
                        <span style='color:{perf_color(lstm['R2'], gb['R2'], "r2")}; font-weight:bold;'> {lstm['R2']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("🔍 Please enter a stock ticker to begin.")

st.markdown("""
    <hr style="margin-top: 40px; margin-bottom: 10px; border: 0; border-top: 1px solid #555;">
    <div style='font-size: 13px; color: #999999; text-align: center;'>
        ⚠️ <strong>Disclaimer:</strong> This dashboard is intended for educational and experimental purposes only.
        The forecasts and outputs generated by this tool are based on historical data, machine learning models, and
        simulated results, and should not be interpreted as financial advice or recommendations for trading.
        Please consult with a qualified financial advisor before making any investment decisions.
    </div>
""", unsafe_allow_html=True)
