import pandas as pd


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


def calculate_trading_plan(
    hist_df: pd.DataFrame,
    current_price: float,
    fair_value: float,
    capital_rp: float = 10_000_000.0,
    max_risk_pct: float = 2.0,
) -> dict:
    """Menyusun Trading Plan objektif (Support, Resistance, ATR Stop Loss, R:R & Position Sizing Lot)."""
    df = hist_df.copy()

    # Hitung ATR-14 (Average True Range) untuk mengukur volatilitas harian normal
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1])

    # Level Support & Resistance Historis
    sup_20 = float(df["Low"].iloc[-20:].min())
    sup_60 = float(df["Low"].iloc[-60:].min())
    res_20 = float(df["High"].iloc[-20:].max())
    res_60 = float(df["High"].iloc[-60:].max())

    # Area Beli (Entry Ideal dekat Support 20D atau harga saat ini)
    entry_ideal = round_idx_tick(max(sup_20, current_price - 0.5 * atr_14))
    entry_now = round_idx_tick(current_price)

    # Stop Loss berbasis 1.5x ATR di bawah harga masuk (dibatasi maksimal -8% agar ketat)
    raw_sl = min(entry_now - 1.5 * atr_14, sup_20 * 0.985)
    raw_sl = max(raw_sl, entry_now * 0.92)
    stop_loss = round_idx_tick(raw_sl)
    if stop_loss >= entry_now:
        stop_loss = round_idx_tick(entry_now * 0.97)

    # Target Profit 1 (Resistance Terdekat) & Target Profit 2 (Resistance 60D / Harga Wajar)
    raw_tp1 = res_20 if res_20 > entry_now * 1.02 else (entry_now + 2.0 * atr_14)
    tp1 = round_idx_tick(raw_tp1)

    candidates_tp2 = [res_60, entry_now + 3.5 * atr_14]
    if fair_value > entry_now * 1.05:
        candidates_tp2.append(fair_value)
    tp2 = round_idx_tick(max(candidates_tp2))
    if tp2 <= tp1:
        tp2 = round_idx_tick(tp1 + 1.5 * atr_14)

    # Hitung Risk & Reward
    risk_per_share = max(entry_now - stop_loss, 1)
    reward1_per_share = max(tp1 - entry_now, 0)
    reward2_per_share = max(tp2 - entry_now, 0)

    risk_pct_trade = (risk_per_share / entry_now) * 100
    reward1_pct = (reward1_per_share / entry_now) * 100
    reward2_pct = (reward2_per_share / entry_now) * 100

    rr_ratio_1 = reward1_per_share / risk_per_share
    rr_ratio_2 = reward2_per_share / risk_per_share

    # Position Sizing (1 Lot = 100 Lembar)
    max_loss_rp = capital_rp * (max_risk_pct / 100.0)
    max_shares_by_risk = int(max_loss_rp // risk_per_share)
    max_shares_by_cap = int(capital_rp // entry_now) if entry_now > 0 else 0
    recommended_shares = min(max_shares_by_risk, max_shares_by_cap)
    recommended_lots = max(0, recommended_shares // 100)

    total_buy_rp = recommended_lots * 100 * entry_now
    est_loss_if_sl_rp = recommended_lots * 100 * risk_per_share
    est_profit_tp1_rp = recommended_lots * 100 * reward1_per_share

    # Penilaian Kelayakan Setup menurut Aturan Manajemen Risiko
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
