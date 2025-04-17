
# 📊 SmartAlpha: Machine Learning-Driven Stock Selection & Portfolio Optimization

SmartAlpha is an end-to-end quantitative equity strategy framework that leverages both traditional financial statement analysis and modern machine learning techniques. It integrates long-term fundamentals with short-term price movements and news sentiment to evaluate stock quality and construct a dynamic, risk-adjusted portfolio.

---

## 🔧 Technical Stack

- **Languages**: Python  
- **Libraries**: `Pandas`, `NumPy`, `scikit-learn`, `XGBoost`, `yfinance`, `requests`, `transformers`, `NLTK` (VADER)  
- **APIs**:
  - [Yahoo Finance](https://finance.yahoo.com/) (historical market data)
  - [Financial Modeling Prep (FMP)](https://site.financialmodelingprep.com/) (fundamentals, financial statements)
  - [Finnhub](https://finnhub.io/) (news and headlines for sentiment)

---

## 💡 Project Objectives

1. **Stock Evaluation Engine**
   - Ingests 5 years of **income statement**, **balance sheet**, and **cash flow** data
   - Trains a **classification model** to determine whether a stock is fundamentally strong and worth buying

2. **Short-Term Signal Modeling**
   - Extracts **technical indicators**, **returns**, and **volatility metrics**
   - Uses **NLP sentiment analysis** (VADER, FinBERT) on company news headlines
   - Applies **XGBoost classifiers** to predict short-term price direction (5–30 day horizon)

3. **Portfolio Optimization**
   - Constructs a **risk-adjusted portfolio** using confidence-weighted allocations
   - Simulates portfolio returns across multiple holding periods (5, 20, 40, 60 days)
   - Calculates **Sharpe Ratio**, volatility, and expected return

---

## 📁 Key Components

- `data/`: Stores historical JSON data, fundamentals, and sentiment  
- `train_fundamentals_model.py`: Trains model on 5Y financial statements to classify "Buy" vs "Avoid"  
- `sentiment_analysis.py`: Collects and scores news headlines via VADER/FinBERT  
- `portfolio_allocator.py`: Builds 60/40 or dynamic allocation portfolios and simulates performance  
- `fmp_fundamentals.py`: Extracts ratios and valuation data from FMP API  
- `stock_labels.csv`: Custom labels for training ML models on fundamentals

---

## 🔍 Sample Features Used

**From Fundamentals:**
- Revenue growth, EBITDA margin, Net margin
- Free Cash Flow (FCF) average & growth
- Debt-to-equity, Current ratio

**From Price/Sentiment:**
- Lagged returns, Volatility, Momentum
- Sentiment scores from VADER and FinBERT
- Moving averages, Signal confidence

---

## 🧠 ML Models Used

- **RandomForestClassifier** for long-term value classification  
- **XGBoostClassifier** for short-term directional predictions  
- Feature scaling with `StandardScaler`  
- Feature importance visualizations

---

## 📈 Results & Metrics

- High signal accuracy on fundamental classification (AUC ≥ 0.85 on test set)
- Portfolio Sharpe Ratios: Up to **6.5+** depending on asset mix and prediction strength
- Configurable horizon testing: 5D, 20D, 40D, 60D

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/SmartAlpha.git
cd SmartAlpha

# Install dependencies
pip install -r requirements.txt

# Run data collection
python download_historical_data.py
python fmp_fundamentals.py

# Train model
python train_fundamentals_model.py

# Build a portfolio
python portfolio_allocator.py
