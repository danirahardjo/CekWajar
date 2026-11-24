import pandas as pd
import streamlit as st
import yfinance as yf


def _extract_fundamental_fallback(ticker: yf.Ticker, info: dict) -> dict:
    """Mengambil EPS dan BVPS dari laporan keuangan jika endpoint ticker.info dibatasi Yahoo Finance."""
    enriched = dict(info)

    # 1. Fallback EPS dari Laporan Laba Rugi (Kuartalan disetahunkan / Tahunan)
    if not enriched.get("trailingEps"):
        try:
            q_inc = ticker.quarterly_income_stmt
            if q_inc is not None and not q_inc.empty:
                for row_name in ["Diluted EPS", "Basic EPS"]:
                    if row_name in q_inc.index:
                        vals = q_inc.loc[row_name].dropna().iloc[:4]
                        if len(vals) > 0:
                            ttm_eps = float(vals.sum()) * (4.0 / len(vals))
                            enriched["trailingEps"] = round(ttm_eps, 2)
                            break
        except Exception:
            pass

    if not enriched.get("trailingEps"):
        try:
            a_inc = ticker.income_stmt
            if a_inc is not None and not a_inc.empty:
                for row_name in ["Diluted EPS", "Basic EPS"]:
                    if row_name in a_inc.index:
                        vals = a_inc.loc[row_name].dropna()
                        if len(vals) > 0:
                            enriched["trailingEps"] = round(float(vals.iloc[0]), 2)
                            break
        except Exception:
            pass

    # 2. Fallback BVPS (Total Ekuitas / Jumlah Saham Beredar) dari Neraca
    if not enriched.get("bookValue"):
        try:
            bs = ticker.quarterly_balance_sheet
            if bs is None or bs.empty:
                bs = ticker.balance_sheet

            if bs is not None and not bs.empty:
                equity = None
                for eq_row in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                    if eq_row in bs.index:
                        eq_vals = bs.loc[eq_row].dropna()
                        if len(eq_vals) > 0:
                            equity = float(eq_vals.iloc[0])
                            break

                shares = None
                for sh_row in ["Ordinary Shares Number", "Share Issued"]:
                    if sh_row in bs.index:
                        sh_vals = bs.loc[sh_row].dropna()
                        if len(sh_vals) > 0:
                            shares = float(sh_vals.iloc[0])
                            break

                if not shares:
                    try:
                        shares = float(ticker.fast_info.get("shares") or 0)
                    except Exception:
                        shares = None

                if equity and shares and shares > 0:
                    enriched["bookValue"] = round(equity / shares, 2)
        except Exception:
            pass

    return enriched


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

    info = _extract_fundamental_fallback(ticker, info)

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
