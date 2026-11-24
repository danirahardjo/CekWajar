import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=1800)
def fetch_stock_data(ticker_symbol: str, period: str = "2y"):
    """Mengambil data historis harian & fundamental dari Yahoo Finance untuk kode saham IDX."""
    clean_ticker = ticker_symbol.strip().upper()
    if not clean_ticker.endswith(".JK"):
        clean_ticker = f"{clean_ticker}.JK"

    ticker = yf.Ticker(clean_ticker)
    hist = ticker.history(period=period)

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist = hist.dropna(subset=["Close"]).copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    return clean_ticker, hist, info


@st.cache_data(ttl=45)
def fetch_intraday_volume_data(clean_ticker: str, interval: str = "5m") -> pd.DataFrame:
    """Mengambil data intraday terbaru (WIB) untuk mendeteksi lonjakan volume jual/beli masif."""
    period_map = {"1m": "1d", "5m": "5d", "15m": "5d"}
    period = period_map.get(interval, "5d")

    ticker = yf.Ticker(clean_ticker)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close", "Volume"]).copy()
    df = df[df["Volume"] > 0].copy()
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jakarta")
    else:
        df.index = df.index.tz_convert("Asia/Jakarta")

    return df
