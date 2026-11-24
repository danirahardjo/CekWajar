import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def round_idx_tick(price: float) -> int:
    """Membulatkan harga sesuai aturan fraksi harga (tick size) Bursa Efek Indonesia (BEI)."""
    if price <= 0:
        return 0
    if price < 200:
        tick = 1
    elif price < 500:
        tick = 2
    elif price < 2000:
        tick = 5
    elif price < 5000:
        tick = 10
    else:
        tick = 25
    return int(round(price / tick) * tick)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_ihsg_close(period: str = "2y") -> pd.Series:
    """Mengambil data penutupan IHSG (^JKSE) untuk mengukur kekuatan relatif saham terhadap pasar."""
    try:
        ihsg = yf.Ticker("^JKSE").history(period=period)
        if ihsg is not None and not ihsg.empty:
            if isinstance(ihsg.columns, pd.MultiIndex):
                ihsg.columns = ihsg.columns.get_level_values(0)
            s = ihsg["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
    except Exception:
        pass
    return pd.Series(dtype=float)


def _compute_technical_and_smart_money(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung indikator tren (EMA, MACD, RSI) dan arus dana harian (Chaikin Money Flow 20D)."""
    df = hist_df.copy()

    # Moving Averages (EMA)
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, min_periods=20, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI-14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # Chaikin Money Flow 20 Hari (Akumulasi vs Distribusi Institusi Harian)
    hl_range = df["High"] - df["Low"]
    mfm = np.where(
        hl_range > 0,
        ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range,
        0.0,
    )
    mfv = mfm * df["Volume"]
    vol_sum_20 = df["Volume"].rolling(20, min_periods=5).sum().replace(0, np.nan)
    df["CMF_20"] = (pd.Series(mfv, index=df.index).rolling(20, min_periods=5).sum() / vol_sum_20).fillna(0.0)

    return df


def _run_signal_backtest(tech_df: pd.DataFrame, hold_days: int = 10) -> dict:
    """Menguji win-rate historis sinyal masuk (Tren Naik + Akumulasi CMF + RSI Sehat) pada saham ini."""
    if len(tech_df) < 60:
        return {"total_signals": 0, "win_rate": 0.0, "avg_return": 0.0}

    df = tech_df.copy()
    df["Fwd_Return"] = (df["Close"].shift(-hold_days) / df["Close"]) - 1.0

    raw_cond = (
        (df["Close"] > df["EMA_20"])
        & (df["EMA_20"] > df["EMA_50"])
        & (df["CMF_20"] > 0.03)
        & (df["RSI_14"] >= 45)
        & (df["RSI_14"] <= 70)
    )
    # Ambil hanya titik awal munculnya sinyal (bukan hari berturut-turut)
    entry_signals = raw_cond & (~raw_cond.shift(1).fillna(False))
    valid_trades = df[entry_signals].dropna(subset=["Fwd_Return"])

    total = len(valid_trades)
    if total == 0:
        return {"total_signals": 0, "win_rate": 0.0, "avg_return": 0.0}

    wins = int((valid_trades["Fwd_Return"] > 0).sum())
    win_rate = (wins / total) * 100.0
    avg_return = float(valid_trades["Fwd_Return"].mean()) * 100.0

    return {
        "total_signals": total,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


def analyze_entry_conviction(
    hist_df: pd.DataFrame,
    valuation: dict,
    piotroski: dict,
    composite_sentiment: float,
    period: str = "2y",
) -> dict:
    """Menghitung Skor Keyakinan Masuk (0-100) dari 4 pilar + checklist + backtest historis."""
    tech_df = _compute_technical_and_smart_money(hist_df)
    last = tech_df.iloc[-1]
    close_now = float(last["Close"])
    ema20 = float(last["EMA_20"])
    ema50 = float(last["EMA_50"])
    ema200 = float(last["EMA_200"])
    macd_hist = float(last["MACD_Hist"])
    rsi14 = float(last["RSI_14"])
    cmf20 = float(last["CMF_20"])

    # Kekuatan Relatif vs IHSG (20 Hari Bursa)
    ihsg_close = _fetch_ihsg_close(period=period)
    alpha_vs_ihsg = 0.0
    has_ihsg = False
    if len(tech_df) >= 21 and len(ihsg_close) >= 21:
        aligned = pd.DataFrame({"stock": tech_df["Close"], "ihsg": ihsg_close}).dropna()
        if len(aligned) >= 21:
            stock_ret_20 = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[-21]) - 1.0
            ihsg_ret_20 = (aligned["ihsg"].iloc[-1] / aligned["ihsg"].iloc[-21]) - 1.0
            alpha_vs_ihsg = (stock_ret_20 - ihsg_ret_20) * 100.0
            has_ihsg = True

    # ------------------------------------------
    # PILAR 1: VALUASI & MARGIN OF SAFETY (30%)
    # ------------------------------------------
    fv = float(valuation.get("composite") or 0.0)
    mos_pct = float(valuation.get("mos_pct") or 0.0)
    if fv > 0:
        score_val = float(np.clip(((mos_pct + 20.0) / 55.0) * 100.0, 0.0, 100.0))
    else:
        score_val = 35.0

    # ------------------------------------------
    # PILAR 2: KESEHATAN FUNDAMENTAL F-SCORE (25%)
    # ------------------------------------------
    fscore_9 = int(piotroski.get("score_9") or 0)
    max_ev = int(piotroski.get("max_evaluable") or 0)
    score_fund = (fscore_9 / 9.0) * 100.0 if max_ev > 0 else 50.0

    # ------------------------------------------
    # PILAR 3: TREN TEKNIKAL & ANTI-FALLING KNIFE (25%)
    # ------------------------------------------
    trend_pts = 0.0
    if close_now > ema20:
        trend_pts += 30.0
    if ema20 > ema50:
        trend_pts += 25.0
    if close_now > ema200:
        trend_pts += 15.0
    if macd_hist > 0:
        trend_pts += 15.0
    if 45.0 <= rsi14 <= 72.0:
        trend_pts += 15.0
    elif 35.0 <= rsi14 < 45.0:
        trend_pts += 7.0
    score_tech = min(trend_pts, 100.0)

    # ------------------------------------------
    # PILAR 4: SMART MONEY (CMF-20), ALPHA IHSG & SENTIMEN (20%)
    # ------------------------------------------
    cmf_pts = float(np.clip(((cmf20 + 0.15) / 0.30) * 55.0, 0.0, 55.0))
    alpha_pts = float(np.clip(((alpha_vs_ihsg + 5.0) / 10.0) * 25.0, 0.0, 25.0)) if has_ihsg else 12.5
    sent_pts = float(np.clip(((composite_sentiment + 0.4) / 0.8) * 20.0, 0.0, 20.0))
    score_flow = min(cmf_pts + alpha_pts + sent_pts, 100.0)

    # SKOR KEYAKINAN TOTAL (0-100)
    conviction_score = round(
        (score_val * 0.30) + (score_fund * 0.25) + (score_tech * 0.25) + (score_flow * 0.20),
        1,
    )

    # Checklist Syarat Masuk Objektif
    checklist = [
        {
            "Pilar": "1. Valuasi",
            "Syarat Konfirmasi Masuk": f"Harga Diskon / Margin of Safety ≥ 15% (Saat ini: {mos_pct:+.1f}%)" if fv > 0 else "Data Valuasi Wajar Tersedia",
            "Status": "✅ Terpenuhi" if (fv > 0 and mos_pct >= 15.0) else "⏳ Belum Terpenuhi",
        },
        {
            "Pilar": "2. Kualitas",
            "Syarat Konfirmasi Masuk": f"Laporan Keuangan Sehat / F-Score ≥ 7/9 (Saat ini: {fscore_9}/9)" if max_ev > 0 else "Skor Kesehatan Laporan Keuangan",
            "Status": "✅ Terpenuhi" if (max_ev > 0 and fscore_9 >= 7) else "⏳ Belum Terpenuhi",
        },
        {
            "Pilar": "3. Anti-Falling Knife",
            "Syarat Konfirmasi Masuk": f"Harga Bertahan di Atas EMA-20 (Rp {close_now:,.0f} vs EMA-20 Rp {ema20:,.0f})",
            "Status": "✅ Terpenuhi" if close_now > ema20 else "⏳ Tunggu Reversal",
        },
        {
            "Pilar": "3. Tren Menengah",
            "Syarat Konfirmasi Masuk": f"Struktur Uptrend EMA-20 > EMA-50 & RSI Sehat (RSI: {rsi14:.1f})",
            "Status": "✅ Terpenuhi" if (ema20 > ema50 and 45.0 <= rsi14 <= 75.0) else "⏳ Belum Terpenuhi",
        },
        {
            "Pilar": "4. Smart Money",
            "Syarat Konfirmasi Masuk": f"Arus Dana Institusi 20D Positif (CMF-20: {cmf20:+.2f})",
            "Status": "✅ Terpenuhi" if cmf20 > 0.03 else "⏳ Belum Terpenuhi",
        },
        {
            "Pilar": "4. Kekuatan Pasar",
            "Syarat Konfirmasi Masuk": f"Mengungguli Kinerja IHSG 20 Hari ({alpha_vs_ihsg:+.2f}%)" if has_ihsg else "Kekuatan Relatif vs IHSG",
            "Status": "✅ Terpenuhi" if (has_ihsg and alpha_vs_ihsg > 0) else "⏳ Belum Terpenuhi",
        },
    ]

    passed_count = sum(1 for item in checklist if "✅" in item["Status"])

    if conviction_score >= 72 and close_now > ema20:
        action_label = "HIGH CONVICTION BUY (Layak Masuk Bertahap) 🟢"
    elif conviction_score >= 60 and close_now <= ema20:
        action_label = "WATCHLIST / TUNGGU KONFIRMASI REVERSAL (Murah, Tunggu Tembus EMA-20) 🟡"
    elif conviction_score >= 55:
        action_label = "MODERAT / SPECULATIVE BUY (Gunakan Ukuran Lot Kecil) 🟡"
    else:
        action_label = "HOLD / HINDARI DULU (Konfirmasi Belum Mendukung) 🔴"

    backtest = _run_signal_backtest(tech_df, hold_days=10)

    return {
        "conviction_score": conviction_score,
        "action_label": action_label,
        "passed_count": passed_count,
        "total_checks": len(checklist),
        "pillar_scores": {
            "Valuasi (30%)": round(score_val, 1),
            "Kesehatan F-Score (25%)": round(score_fund, 1),
            "Tren & Reversal (25%)": round(score_tech, 1),
            "Smart Money & Sentimen (20%)": round(score_flow, 1),
        },
        "metrics": {
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "rsi14": rsi14,
            "cmf20": cmf20,
            "alpha_vs_ihsg": alpha_vs_ihsg,
            "has_ihsg": has_ihsg,
        },
        "checklist": checklist,
        "backtest": backtest,
    }


def calculate_trading_plan(
    hist_df: pd.DataFrame,
    current_price: float,
    fair_value: float,
    capital_rp: float = 10_000_000.0,
    max_risk_pct: float = 2.0,
) -> dict:
    """Menyusun Trading Plan objektif (Support, Resistance, ATR Stop Loss, R:R & Position Sizing Lot)."""
    df = hist_df.copy()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1])

    sup_20 = float(df["Low"].iloc[-20:].min())
    sup_60 = float(df["Low"].iloc[-60:].min())
    res_20 = float(df["High"].iloc[-20:].max())
    res_60 = float(df["High"].iloc[-60:].max())

    entry_ideal = round_idx_tick(max(sup_20, current_price - 0.5 * atr_14))
    entry_now = round_idx_tick(current_price)

    raw_sl = min(entry_now - 1.5 * atr_14, sup_20 * 0.985)
    raw_sl = max(raw_sl, entry_now * 0.92)
    stop_loss = round_idx_tick(raw_sl)
    if stop_loss >= entry_now:
        stop_loss = round_idx_tick(entry_now * 0.97)

    raw_tp1 = res_20 if res_20 > entry_now * 1.02 else (entry_now + 2.0 * atr_14)
    tp1 = round_idx_tick(raw_tp1)

    candidates_tp2 = [res_60, entry_now + 3.5 * atr_14]
    if fair_value > entry_now * 1.05:
        candidates_tp2.append(fair_value)
    tp2 = round_idx_tick(max(candidates_tp2))
    if tp2 <= tp1:
        tp2 = round_idx_tick(tp1 + 1.5 * atr_14)

    risk_per_share = max(entry_now - stop_loss, 1)
    reward1_per_share = max(tp1 - entry_now, 0)
    reward2_per_share = max(tp2 - entry_now, 0)

    risk_pct_trade = (risk_per_share / entry_now) * 100
    reward1_pct = (reward1_per_share / entry_now) * 100
    reward2_pct = (reward2_per_share / entry_now) * 100

    rr_ratio_1 = reward1_per_share / risk_per_share
    rr_ratio_2 = reward2_per_share / risk_per_share

    max_loss_rp = capital_rp * (max_risk_pct / 100.0)
    max_shares_by_risk = int(max_loss_rp // risk_per_share)
    max_shares_by_cap = int(capital_rp // entry_now) if entry_now > 0 else 0
    recommended_shares = min(max_shares_by_risk, max_shares_by_cap)
    recommended_lots = max(0, recommended_shares // 100)

    total_buy_rp = recommended_lots * 100 * entry_now
    est_loss_if_sl_rp = recommended_lots * 100 * risk_per_share
    est_profit_tp1_rp = recommended_lots * 100 * reward1_per_share

    if rr_ratio_1 >= 2.0:
        setup_verdict = f"SANGAT MENARIK (R:R 1 : {rr_ratio_1:.2f}) 🟢"
    elif rr_ratio_1 >= 1.2:
        setup_verdict = f"LAYAK DIPANTAU (R:R 1 : {rr_ratio_1:.2f}) 🟡"
    else:
        setup_verdict = f"KURANG IDEAL / DEKAT RESISTANCE (R:R 1 : {rr_ratio_1:.2f}) 🔴"

    return {
        "atr_14": atr_14,
        "sup_20": round_idx_tick(sup_20),
        "sup_60": round_idx_tick(sup_60),
        "res_20": round_idx_tick(res_20),
        "res_60": round_idx_tick(res_60),
        "entry_now": entry_now,
        "entry_ideal": entry_ideal,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct_trade": risk_pct_trade,
        "reward1_pct": reward1_pct,
        "reward2_pct": reward2_pct,
        "rr_ratio_1": rr_ratio_1,
        "rr_ratio_2": rr_ratio_2,
        "recommended_lots": recommended_lots,
        "total_buy_rp": total_buy_rp,
        "est_loss_if_sl_rp": est_loss_if_sl_rp,
        "est_profit_tp1_rp": est_profit_tp1_rp,
        "setup_verdict": setup_verdict,
    }
