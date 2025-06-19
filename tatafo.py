# chatbot_app.py

import streamlit as st
import json
import os
import re
import requests
import datetime
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import nltk
nltk.data.path.append("nltk_data")  # Ensures it uses local path

from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from difflib import get_close_matches

# === NLTK Setup (cached) ===
NLTK_PATH = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(NLTK_PATH)

REQUIRED_NLTK = {
    'punkt': 'tokenizers/punkt',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
    'words': 'corpora/words'
}

def ensure_nltk_data():
    for pkg, path in REQUIRED_NLTK.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_PATH)

# === Entity Detection ===
def extract_named_entities(text):
    entities = []
    for sent in nltk.sent_tokenize(text):
        chunks = ne_chunk(pos_tag(word_tokenize(sent)))
        for chunk in chunks:
            if isinstance(chunk, Tree):
                name = " ".join(c[0] for c in chunk.leaves())
                entities.append(name)
    return entities

# === Load/Save Data ===
def load_data():
    if os.path.exists("chatbot_data.json"):
        with open("chatbot_data.json", "r") as f:
            return json.load(f)
    return {"history": [], "custom_responses": {}}

def save_data(data):
    with open("chatbot_data.json", "w") as f:
        json.dump(data, f, indent=4)

def get_default_responses():
    return {
        "hi": "Hello! How can I assist you?",
        "hello": "Hi there! What can I help you with?",
        "bye": "Goodbye! Come back soon.",
        "what is python": "Python is a versatile programming language used in many fields.",
        "how are you": "I'm a chatbot — always operational!",
        "what programming language are you learning?": "Python.",
        "where are you learning data science": "Sail Innovation Lab."
    }

# === API and Info Functions ===
def get_weather(text, key):
    if not key:
        return "No weather API key set."
    match = re.search(r"weather in ([a-zA-Z ]+)", text)
    city = match.group(1) if match else "London"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    try:
        res = requests.get(url).json()
        if res.get("main"):
            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]
            return f"Weather in {city.title()}: {temp}°C, {desc}"
        return f"Could not retrieve weather for {city}."
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"

def get_news(key):
    if not key:
        return "No news API key set."
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={key}"
    try:
        res = requests.get(url).json()
        articles = res.get("articles", [])[:3]
        return "\n\n".join([f"- {a['title']} ({a['source']['name']})" for a in articles])
    except Exception as e:
        return f"News lookup failed: {str(e)}"

def get_wikipedia_summary(query):
    try:
        query = query.lower().strip()
        query = re.sub(r"^(who|what|where|when|how)( is| was| are| does| do| did)?", "", query).strip()
        query = query if query else "Python (programming language)"
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        res = requests.get(url).json()
        if "title" in res and res["title"].lower() == "not found":
            return f"No Wikipedia info for '{query}'."
        if "extract" in res:
            return res["extract"]
        return "Wikipedia summary not found."
    except Exception as e:
        return f"Wikipedia lookup failed: {str(e)}"

def get_crypto_price(text, key):
    if not key:
        return "No crypto API key set."
    match = re.search(r"\b(bitcoin|btc|eth|ethereum|doge)\b", text)
    symbol = match.group(1).upper() if match else "BTC"
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
    try:
        res = requests.get(url).json()
        price = res[symbol.lower()]["usd"]
        return f"Current price of {symbol}: ${price}"
    except:
        return "Crypto lookup failed."

def get_stock_price(text, key):
    if not key:
        return "No stock API key set."
    match = re.search(r"stock (price )?of ([A-Za-z]+)", text)
    symbol = match.group(2).upper() if match else "AAPL"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={key}"
    try:
        res = requests.get(url).json()
        price = res["Global Quote"]["05. price"]
        return f"Current price of {symbol}: ${price}"
    except:
        return "Stock lookup failed."

def get_sports_info(text, key):
    if not key:
        return "No sports API key set."
    return f"(Demo) Sports info for: {text}"

# === CSV Upload & Charting ===
def handle_csv_upload():
    uploaded_file = st.sidebar.file_uploader("Upload stock CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded and loaded.")
        return df
    if st.sidebar.button("Fetch Sample CSV"):
        url = "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1663372800&period2=1694908800&interval=1d&events=history"
        try:
            res = requests.get(url)
            df = pd.read_csv(BytesIO(res.content))
            st.success("Sample CSV fetched.")
            return df
        except:
            st.error("Sample CSV fetch failed.")
    return None

def plot_stock_chart(df):
    st.subheader("Stock Price Chart")
    if df is not None:
        start = st.date_input("Start Date", value=pd.to_datetime(df['Date'].min()))
        end = st.date_input("End Date", value=pd.to_datetime(df['Date'].max()))
        if start <= end:
            mask = (pd.to_datetime(df['Date']) >= start) & (pd.to_datetime(df['Date']) <= end)
            filtered = df.loc[mask]
            fig, ax = plt.subplots()
            ax.plot(filtered['Date'], filtered['Close'], label='Close Price')
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.set_title("Stock Price Over Time")
            st.pyplot(fig)
        else:
            st.warning("Start date must be before end date.")

# === Response Generator ===
def generate_response(user_input, data, api_keys):
    text = user_input.lower().strip()
    tokens = []
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()
    for sent in sent_tokenizer.tokenize(text):
        tokens.extend(word_tokenizer.tokenize(sent))

    matched = get_close_matches(
        text,
        list(data["custom_responses"].keys()) + list(get_default_responses().keys()),
        n=1,
        cutoff=0.6
    )
    if matched:
        if matched[0] in data["custom_responses"]:
            return data["custom_responses"][matched[0]]
        return get_default_responses()[matched[0]]

    if "weather" in tokens:
        return get_weather(text, api_keys.get("weather"))
    elif "news" in tokens:
        return get_news(api_keys.get("news"))
    elif any(w in tokens for w in ["price", "crypto", "bitcoin"]):
        return get_crypto_price(text, api_keys.get("crypto"))
    elif any(w in tokens for w in ["stock", "shares", "company"]):
        return get_stock_price(text, api_keys.get("stock"))
    elif any(w in tokens for w in ["ronaldo", "messi", "match", "haaland", "lebron"]):
        return get_sports_info(text, api_keys.get("sports"))

    entities = extract_named_entities(user_input)
    if entities:
        return get_wikipedia_summary(entities[0])
    return get_wikipedia_summary(user_input)

# === Main Streamlit App ===
def main():
    ensure_nltk_data()

    st.set_page_config(page_title="ChatBot App", layout="centered")
    st.title("Tatafo with AI 🗣️")

    st.sidebar.title("🔐 API Input")
    api_keys = {}
    data = load_data()

    mode = st.sidebar.radio("Provide API keys:", ["Load from file", "Manual input", "Skip"])
    if mode == "Load from file":
        uploaded = st.sidebar.file_uploader("Upload .txt with API keys (key=value per line)")
        if uploaded:
            content = uploaded.read().decode("utf-8").splitlines()
            for line in content:
                if "=" in line:
                    k, v = line.strip().split("=")
                    api_keys[k.strip()] = v.strip()
            st.sidebar.success("✅ API keys loaded.")
    elif mode == "Manual input":
        api_keys["weather"] = st.sidebar.text_input("Weather API Key")
        api_keys["news"] = st.sidebar.text_input("News API Key")
        api_keys["crypto"] = st.sidebar.text_input("Crypto API Key")
        api_keys["stock"] = st.sidebar.text_input("Stock API Key")
        api_keys["sports"] = st.sidebar.text_input("Sports API Key")

        if st.sidebar.button("Save to .txt"):
            with open("api_keys.txt", "w") as f:
                for k, v in api_keys.items():
                    f.write(f"{k}={v}\n")
            st.sidebar.success("Saved to api_keys.txt")

    user_input = st.text_input("You:")
    if user_input:
        response = generate_response(user_input, data, api_keys)
        data["history"].append({"user": user_input, "bot": response})
        save_data(data)
        st.write(f"**Bot:** {response}")

    # Custom response
    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ Add Custom Response")
    new_key = st.sidebar.text_input("Trigger")
    new_resp = st.sidebar.text_area("Bot Response")
    if st.sidebar.button("Add Response"):
        if new_key and new_resp:
            data["custom_responses"][new_key.lower()] = new_resp
            save_data(data)
            st.sidebar.success("Custom response added!")

    if st.sidebar.button("🗑️ Clear Chat History"):
        data["history"] = []
        save_data(data)
        st.sidebar.success("Chat history cleared.")

    if st.sidebar.checkbox("📓 Show Chat History"):
        st.sidebar.markdown("---")
        for entry in data["history"]:
            st.sidebar.write(f"**You:** {entry['user']}")
            st.sidebar.write(f"**Bot:** {entry['bot']}")

    df_stock = handle_csv_upload()
    if df_stock is not None:
        plot_stock_chart(df_stock)

if __name__ == "__main__":
    main()
