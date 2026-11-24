import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600)
def _get_usdidr_rate() -> float:
    """Mengambil kurs terkini USD ke IDR dari Yahoo Finance dengan cadangan default."""
    try:
        fx = yf.Ticker("USDIDR=X").history(period="5d")
        if not fx.empty:
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 1000:
                return rate
    except Exception:
        pass
    return 16200.0


def _extract_fundamental_fallback(ticker: yf.Ticker, info: dict, hist: pd.DataFrame) -> dict:
    """Mengambil metrik fundamental lengkap dari laporan keuangan serta menyelaraskan mata uang ke IDR."""
    enriched = dict(info)
    current_price = float(enriched.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else 0.0))

    q_inc = None
    a_inc = None
    bs = None
    cf = None

    try:
        q_inc = ticker.quarterly_income_stmt
    except Exception:
        pass
    try:
        a_inc = ticker.income_stmt
    except Exception:
        pass
    try:
        bs = ticker.quarterly_balance_sheet
        if bs is None or bs.empty:
            bs = ticker.balance_sheet
    except Exception:
        pass
    try:
        cf = ticker.cash_flow
        if cf is None or cf.empty:
            cf = ticker.quarterly_cash_flow
    except Exception:
        pass

    # 1. Fallback EPS (pertahankan presisi penuh sebelum konversi kurs)
    if not enriched.get("trailingEps"):
        if q_inc is not None and not q_inc.empty:
            for row_name in ["Diluted EPS", "Basic EPS"]:
                if row_name in q_inc.index:
                    vals = q_inc.loc[row_name].dropna().iloc[:4]
                    if len(vals) > 0:
                        enriched["trailingEps"] = float(vals.sum()) * (4.0 / len(vals))
                        break
        if not enriched.get("trailingEps") and a_inc is not None and not a_inc.empty:
            for row_name in ["Diluted EPS", "Basic EPS"]:
                if row_name in a_inc.index:
                    vals = a_inc.loc[row_name].dropna()
                    if len(vals) > 0:
                        enriched["trailingEps"] = float(vals.iloc[0])
                        break

    # 2. Ekuitas, Jumlah Saham Beredar & BVPS
    equity = None
    shares = None
    if bs is not None and not bs.empty:
        for eq_row in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
            if eq_row in bs.index:
                eq_vals = bs.loc[eq_row].dropna()
                if len(eq_vals) > 0:
                    equity = float(eq_vals.iloc[0])
                    break
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

    if shares and shares > 0:
        enriched["sharesOutstanding"] = shares

    if not enriched.get("bookValue") and equity and shares and shares > 0:
        enriched["bookValue"] = equity / shares

    # 3. Fallback ROE (Net Income / Total Ekuitas)
    if not enriched.get("returnOnEquity") and equity and equity > 0:
        net_inc = None
        if q_inc is not None and not q_inc.empty and "Net Income" in q_inc.index:
            ni_vals = q_inc.loc["Net Income"].dropna().iloc[:4]
            if len(ni_vals) > 0:
                net_inc = float(ni_vals.sum()) * (4.0 / len(ni_vals))
        elif a_inc is not None and not a_inc.empty and "Net Income" in a_inc.index:
            ni_vals = a_inc.loc["Net Income"].dropna()
            if len(ni_vals) > 0:
                net_inc = float(ni_vals.iloc[0])
        if net_inc is not None:
            enriched["returnOnEquity"] = round(net_inc / equity, 4)

    # 4. Pertumbuhan Laba Tahunan (YoY Earnings Growth)
    if not enriched.get("earningsGrowth") and a_inc is not None and not a_inc.empty:
        for row_name in ["Diluted EPS", "Basic EPS", "Net Income"]:
            if row_name in a_inc.index:
                vals = a_inc.loc[row_name].dropna()
                if len(vals) >= 2 and abs(float(vals.iloc[1])) > 0:
                    growth = (float(vals.iloc[0]) - float(vals.iloc[1])) / abs(float(vals.iloc[1]))
                    enriched["earningsGrowth"] = float(np.clip(growth, -0.30, 0.40))
                    break

    # 5. Free Cash Flow per Share & Net Debt per Share (untuk DCF)
    if shares and shares > 0:
        fcf = enriched.get("freeCashflow")
        if fcf is None and cf is not None and not cf.empty:
            if "Free Cash Flow" in cf.index:
                fcf_vals = cf.loc["Free Cash Flow"].dropna()
                if len(fcf_vals) > 0:
                    fcf = float(fcf_vals.iloc[0])
            elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
                ocf = cf.loc["Operating Cash Flow"].dropna()
                capex = cf.loc["Capital Expenditure"].dropna()
                if len(ocf) > 0 and len(capex) > 0:
                    fcf = float(ocf.iloc[0]) - abs(float(capex.iloc[0]))
        if fcf is not None:
            enriched["fcfPerShare"] = float(fcf) / shares

        if bs is not None and not bs.empty:
            total_debt = 0.0
            cash_val = 0.0
            if "Total Debt" in bs.index and not bs.loc["Total Debt"].dropna().empty:
                total_debt = float(bs.loc["Total Debt"].dropna().iloc[0])
            for c_row in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                if c_row in bs.index and not bs.loc[c_row].dropna().empty:
                    cash_val = float(bs.loc[c_row].dropna().iloc[0])
                    break
            enriched["netDebtPerShare"] = (total_debt - cash_val) / shares

    # 6. Deteksi & Konversi Mata Uang Laporan Keuangan (USD -> IDR)
    eps_raw = float(enriched.get("trailingEps") or 0.0)
    bvps_raw = float(enriched.get("bookValue") or 0.0)
    fin_curr = (enriched.get("financialCurrency") or "").upper()

    is_usd_report = fin_curr == "USD"
    # Deteksi otomatis jika ticker.info kosong namun rasio harga/BVPS & harga/EPS tidak wajar (>300x)
    if not fin_curr and current_price >= 50:
        pbv_unrealistic = bvps_raw > 0 and (current_price / bvps_raw) > 300
        per_unrealistic = eps_raw <= 0 or (current_price / eps_raw) > 300
        if pbv_unrealistic and per_unrealistic:
            is_usd_report = True

    if is_usd_report:
        fx_rate = _get_usdidr_rate()
        enriched["financialCurrency"] = f"USD (Kurs Rp {fx_rate:,.0f})"
        if enriched.get("trailingEps"):
            enriched["trailingEps"] = float(enriched["trailingEps"]) * fx_rate
        if enriched.get("bookValue"):
            enriched["bookValue"] = float(enriched["bookValue"]) * fx_rate
        if enriched.get("fcfPerShare"):
            enriched["fcfPerShare"] = float(enriched["fcfPerShare"]) * fx_rate
        if enriched.get("netDebtPerShare"):
            enriched["netDebtPerShare"] = float(enriched["netDebtPerShare"]) * fx_rate

    if enriched.get("trailingEps"):
        enriched["trailingEps"] = round(float(enriched["trailingEps"]), 2)
    if enriched.get("bookValue"):
        enriched["bookValue"] = round(float(enriched["bookValue"]), 2)

    # 7. Median Historis PER & PBV Emiten (setelah penyelarasan kurs)
    if not hist.empty:
        median_hist_price = float(hist["Close"].median())
        eps_now = float(enriched.get("trailingEps") or 0)
        bvps_now = float(enriched.get("bookValue") or 0)
        if eps_now > 0:
            enriched["histMedianPER"] = round(float(np.clip(median_hist_price / eps_now, 5.0, 30.0)), 1)
        if bvps_now > 0:
            enriched["histMedianPBV"] = round(float(np.clip(median_hist_price / bvps_now, 0.5, 6.0)), 1)

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

    info = _extract_fundamental_fallback(ticker, info, hist)

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
