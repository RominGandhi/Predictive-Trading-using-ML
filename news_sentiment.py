import requests
import json
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch
import os

# Load environment variables
load_dotenv("keys.env")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
labels = ['negative', 'neutral', 'positive']

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(probs, dim=1).item()
    sentiment_label = labels[predicted]
    sentiment_score = round(probs[0][2].item() - probs[0][0].item(), 4)  # Positive - Negative
    return sentiment_label, sentiment_score

def run_news_sentiment_pipeline(ticker_input, keyword_input=""):
    BASE_URL = "https://finnhub.io/api/v1/company-news"
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    keyword_filters = [k.strip().lower() for k in keyword_input.split(',') if k.strip()]

    today = datetime.now().date()
    lookback_days = 90
    chunk_size = 30
    chunks = [(today - timedelta(days=i + chunk_size), today - timedelta(days=i)) for i in range(0, lookback_days, chunk_size)]

    # Ensure output directory exists
    output_dir = os.path.join("data", "sentiment")
    os.makedirs(output_dir, exist_ok=True)

    for stock_symbol in tickers:
        print(f"\n📊 Starting sentiment analysis for {stock_symbol}")
        sentiment_articles = []

        for from_date, to_date in chunks:
            print(f"\nFetching news from {from_date} to {to_date}...")

            params = {
                "symbol": stock_symbol,
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY
            }

            response = requests.get(BASE_URL, params=params)
            if response.status_code != 200:
                print(f"❌ API error during {from_date} to {to_date}. Skipping.")
                continue

            data = response.json()
            if not data:
                print("No articles found for this date range.")
                continue

            for article in data:
                title = article.get("headline", "")
                if keyword_filters and not any(k in title.lower() for k in keyword_filters):
                    continue

                source = article.get("source", "Unknown")
                published = datetime.fromtimestamp(article.get("datetime")).strftime('%Y-%m-%d %H:%M:%S')

                sentiment_label, sentiment_score = get_finbert_sentiment(title)

                print(f"Title: {title}")
                print(f"Source: {source}")
                print(f"Published: {published}")
                print(f"Sentiment: {sentiment_label} (score: {sentiment_score:.2f})\n")

                sentiment_articles.append({
                    "title": title,
                    "source": source,
                    "published_at": published,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "matched_keywords": [k for k in keyword_filters if k in title.lower()] if keyword_filters else "None"
                })

        # Save to file
        filename = os.path.join(output_dir, f"{stock_symbol}_news_sentiment.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(sentiment_articles, f, indent=4)

        print(f"\n✅ Total articles collected for {stock_symbol}: {len(sentiment_articles)}")
        print(f"📄 Sentiment data saved to {filename}")
