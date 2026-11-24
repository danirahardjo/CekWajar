from datetime import timedelta, timezone
from email.utils import parsedate_to_datetime
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st

BULLISH_WORDS = {
    # Bahasa Indonesia
    "laba naik": 2.0, "untung": 1.5, "rekor": 1.8, "dividen": 1.7, "ekspansi": 1.5,
    "tumbuh": 1.4, "pertumbuhan": 1.4, "menguat": 1.5, "melonjak": 1.8, "akuisisi": 1.3,
    "buyback": 1.6, "surplus": 1.4, "positif": 1.2, "optimis": 1.3, "rekomendasi beli": 1.8,
    "target harga naik": 1.8, "kinerja cemerlang": 1.9, "pendapatan naik": 1.6, "prospek cerah": 1.5,
    # Bahasa Inggris (Berita Global)
    "surge": 1.8, "soar": 1.8, "profit": 1.5, "growth": 1.4, "dividend": 1.6,
    "beat": 1.6, "outperform": 1.7, "buy": 1.5, "upgrade": 1.8, "bullish": 1.7,
    "expansion": 1.5, "record high": 1.9, "rally": 1.6, "strong": 1.3, "gain": 1.3,
    "rate cut": 1.5, "inflow": 1.5, "recovery": 1.4, "resilient": 1.3,
}

BEARISH_WORDS = {
    # Bahasa Indonesia
    "rugi": -2.0, "kerugian": -2.0, "anjlok": -1.9, "melemah": -1.4, "turun": -1.2,
    "penurunan": -1.3, "utang": -1.2, "gagal bayar": -2.2, "pailit": -2.5, "korupsi": -2.2,
    "sanksi": -1.7, "suspensi": -2.0, "inflasi": -1.2, "suku bunga naik": -1.5, "negatif": -1.3,
    "koreksi": -1.2, "jual bersih": -1.5, "net sell": -1.5, "pangkas": -1.4, "lesu": -1.4,
    # Bahasa Inggris (Berita Global)
    "loss": -1.8, "drop": -1.5, "plunge": -1.9, "fall": -1.3, "decline": -1.4,
    "downgrade": -1.8, "sell": -1.5, "bearish": -1.7, "debt": -1.2, "default": -2.2,
    "lawsuit": -1.7, "miss": -1.6, "weak": -1.4, "recession": -1.8, "inflation": -1.2,
    "rate hike": -1.5, "outflow": -1.5, "slump": -1.7, "risk": -1.1, "warning": -1.5,
}


def analyze_headline_sentiment(text: str) -> float:
    """Menghitung skor sentimen judul berita pada rentang -1.0 (Bearish) hingga +1.0 (Bullish)."""
    if not text:
        return 0.0
    lower_text = text.lower()
    score = 0.0
    for phrase, weight in BULLISH_WORDS.items():
        if phrase in lower_text:
            score += weight
    for phrase, weight in BEARISH_WORDS.items():
        if phrase in lower_text:
            score += weight
    return float(np.tanh(score / 2.5))


def _format_pub_date_wib(pub_date_str: str) -> tuple[str, float]:
    """Mengonversi pubDate RSS menjadi format tanggal saja (contoh: 07 Oct 2026) serta timestamp untuk pengurutan."""
    if not pub_date_str:
        return "-", 0.0
    try:
        dt = parsedate_to_datetime(pub_date_str)
        wib_tz = timezone(timedelta(hours=7))
        dt_wib = dt.astimezone(wib_tz)
        return dt_wib.strftime("%d %b %Y"), dt_wib.timestamp()
    except Exception:
        return pub_date_str[:16], 0.0


def fetch_google_news_rss(query: str, region: str = "ID", max_items: int = 10) -> list[dict]:
    """Mengambil berita terkini dari Google News RSS tanpa memerlukan API Key."""
    encoded_q = urllib.parse.quote(query)
    if region == "ID":
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=id&gl=ID&ceid=ID:id"
    else:
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"

    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for item in root.findall("./channel/item")[:max_items]:
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            pub_date = item.findtext("pubDate", default="")
            source = item.findtext("source", default="Google News")
            formatted_date, timestamp = _format_pub_date_wib(pub_date)
            sentiment = analyze_headline_sentiment(title)
            articles.append({
                "Judul": title,
                "Sumber": source,
                "Tanggal": formatted_date,
                "_ts": timestamp,
                "Skor Sentimen": round(sentiment, 3),
                "Label": "Bullish 🟢" if sentiment > 0.15 else ("Bearish 🔴" if sentiment < -0.15 else "Netral ⚪"),
                "Link": link,
            })
    except Exception:
        pass
    return articles


@st.cache_data(ttl=1800)
def get_dual_market_news(base_code: str, company_name: str):
    """Mengambil berita dalam negeri (Indonesia) dan luar negeri (Global) terurut dari yang terbaru."""
    short_name = company_name.replace("PT ", "").replace(" Tbk", "").strip() if company_name else base_code
    query_id = f'"{base_code}" OR "{short_name}" saham'
    query_global = f'"{short_name}" OR "{base_code}" Indonesia stock OR economy'

    news_id = fetch_google_news_rss(query_id, region="ID", max_items=10)
    news_global = fetch_google_news_rss(query_global, region="US", max_items=10)

    df_id = pd.DataFrame(news_id)
    if not df_id.empty:
        df_id = df_id.sort_values("_ts", ascending=False).drop(columns=["_ts"]).reset_index(drop=True)

    df_global = pd.DataFrame(news_global)
    if not df_global.empty:
        df_global = df_global.sort_values("_ts", ascending=False).drop(columns=["_ts"]).reset_index(drop=True)

    score_id = float(df_id["Skor Sentimen"].mean()) if not df_id.empty else 0.0
    score_global = float(df_global["Skor Sentimen"].mean()) if not df_global.empty else 0.0
    return df_id, df_global, score_id, score_global
