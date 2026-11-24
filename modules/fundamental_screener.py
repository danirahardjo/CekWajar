from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import streamlit as st
import yfinance as yf

from .market_data import _extract_fundamental_fallback
from .valuation import calculate_fair_value


def _analyze_single_ticker_fundamental(
    jk_ticker: str,
    fair_per: float,
    fair_pbv: float,
    discount_rate: float,
) -> dict | None:
    """Menarik data harga & laporan keuangan 1 emiten lalu menghitung Harga Wajar + Piotroski F-Score."""
    try:
        ticker = yf.Ticker(jk_ticker)
        hist = ticker.history(period="1y")
        if hist is None or hist.empty:
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist = hist.dropna(subset=["Close"]).copy()
        if hist.empty:
            return None

        try:
            raw_info = ticker.info or {}
        except Exception:
            raw_info = {}

        info = _extract_fundamental_fallback(ticker, raw_info, hist)

        current_price = float(info.get("currentPrice") or hist["Close"].iloc[-1])
        if current_price <= 0:
            return None

        eps_val = float(info.get("trailingEps") or 0.0)
        bvps_val = float(info.get("bookValue") or 0.0)
        roe_val = float(info.get("returnOnEquity") or 0.0)
        growth_val = float(info.get("earningsGrowth") or 0.0)
        div_yield_val = float(info.get("dividendYield") or 0.0)
        fcf_ps_val = float(info.get("fcfPerShare") or 0.0)
        net_debt_ps_val = float(info.get("netDebtPerShare") or 0.0)
        sector = info.get("sector") or "Tidak Tersedia"
        company_name = info.get("shortName") or info.get("longName") or jk_ticker.replace(".JK", "")

        val_res = calculate_fair_value(
            current_price=current_price,
            eps=eps_val,
            bvps=bvps_val,
            fair_per=fair_per,
            fair_pbv=fair_pbv,
            sector=sector,
            roe=roe_val,
            earnings_growth=growth_val,
            dividend_yield=div_yield_val,
            fcf_per_share=fcf_ps_val,
            net_debt_per_share=net_debt_ps_val,
            discount_rate=discount_rate,
        )

        pio = info.get("piotroski") or {}
        fscore_9 = int(pio.get("score_9", 0))
        max_eval = int(pio.get("max_evaluable", 0))
        pio_verdict = pio.get("verdict", "Data Tidak Tersedia")

        fair_value = float(val_res["composite"])
        mos_pct = float(val_res["mos_pct"]) if fair_value > 0 else -999.0

        per_actual = (current_price / eps_val) if eps_val > 0 else 0.0
        pbv_actual = (current_price / bvps_val) if bvps_val > 0 else 0.0

        # Skor Gabungan (0-100): 55% Diskon Valuasi (MoS) + 45% Kesehatan Fundamental (F-Score)
        mos_component = min(max((mos_pct + 20.0) / 70.0, 0.0), 1.0) * 55.0 if fair_value > 0 else 0.0
        fscore_component = (fscore_9 / 9.0) * 45.0 if max_eval > 0 else 0.0
        quality_value_score = round(mos_component + fscore_component, 1)

        if fair_value > 0 and mos_pct >= 15 and fscore_9 >= 7:
            kuadran = "💎 Zona Emas (Undervalued & Sehat)"
        elif fair_value > 0 and mos_pct >= 15 and fscore_9 < 7:
            kuadran = "⚠️ Waspada Value Trap (Murah, F-Score < 7)"
        elif fscore_9 >= 7:
            kuadran = "🔵 Saham Prima Harga Wajar/Premium"
        else:
            kuadran = "⚪ Netral / Kurang Menarik"

        base_code = jk_ticker.replace(".JK", "")
        return {
            "summary": {
                "Kode": base_code,
                "Emiten": company_name[:28],
                "Sektor": sector,
                "Harga (Rp)": current_price,
                "Harga Wajar (Rp)": fair_value,
                "Margin of Safety (%)": mos_pct if fair_value > 0 else None,
                "F-Score (/9)": fscore_9 if max_eval > 0 else None,
                "Skor Gabungan": quality_value_score,
                "PER (x)": per_actual if per_actual > 0 else None,
                "PBV (x)": pbv_actual if pbv_actual > 0 else None,
                "ROE (%)": (roe_val * 100) if roe_val != 0 else None,
                "Kuadran Analisa": kuadran,
                "Status Valuasi": val_res["status"],
                "Kesehatan Laporan": pio_verdict,
            },
            "detail": {
                "company_name": company_name,
                "sector": sector,
                "current_price": current_price,
                "valuation": val_res,
                "piotroski": pio,
            },
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def scan_fundamental_screener(
    tickers_tuple: tuple[str, ...],
    fair_per: float = 15.0,
    fair_pbv: float = 2.0,
    discount_rate: float = 0.11,
) -> tuple[pd.DataFrame, dict]:
    """Memindai daftar emiten IDX secara paralel untuk valuasi wajar & Piotroski F-Score."""
    clean_tickers = []
    for t in tickers_tuple:
        code = t.strip().upper()
        if code:
            clean_tickers.append(code if code.endswith(".JK") else f"{code}.JK")
    clean_tickers = list(dict.fromkeys(clean_tickers))

    if not clean_tickers:
        return pd.DataFrame(), {}

    rows = []
    details_map = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                _analyze_single_ticker_fundamental,
                jk,
                fair_per,
                fair_pbv,
                discount_rate,
            ): jk
            for jk in clean_tickers
        }
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                code = res["summary"]["Kode"]
                rows.append(res["summary"])
                details_map[code] = res["detail"]

    if not rows:
        return pd.DataFrame(), {}

    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(
        by=["Skor Gabungan", "Margin of Safety (%)"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    return df_res, details_map
