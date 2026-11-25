import numpy as np
import pandas as pd


def analyze_rsi_model(
    hist_df: pd.DataFrame,
    rsi_period: int = 14,
    ma_period: int = 9,
    overbought: int = 70,
    oversold: int = 30,
) -> dict:
    """Menganalisis Wilder's RSI (periode 14), RSI MA-9, Divergence, serta 5 aturan keputusan entry."""
    df = hist_df.copy()

    # 1. Wilder's Smoothed RSI & Moving Average
    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    df["RSI_MA"] = df["RSI"].rolling(ma_period, min_periods=1).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    prev_rsi = df["RSI"].shift(1)
    prev_ma = df["RSI_MA"].shift(1)

    # 2. Deteksi Sinyal Historis Sesuai 5 Aturan Panduan RSI
    # Aturan 1: <= 30 lalu memantul naik (> 30) -> Reversal / Bottom Fishing
    cond_buy_reversal = (prev_rsi <= oversold) & (df["RSI"] > oversold)

    # Aturan 2: 45 - 55 saat harga di atas EMA-20 -> Buy on Weakness
    bow_zone = (df["RSI"] >= 45.0) & (df["RSI"] <= 55.0) & (df["Close"] > df["EMA_20"]) & (df["RSI"] >= prev_rsi)
    cond_buy_bow = bow_zone & (~bow_zone.shift(1).fillna(False))

    # Aturan 3: 55 - 65 dan RSI memotong ke atas MA-9 -> Trend Following
    cond_buy_trend = (
        (df["RSI"] > 55.0)
        & (df["RSI"] <= 65.0)
        & (prev_rsi <= prev_ma)
        & (df["RSI"] > df["RSI_MA"])
    )

    # Aturan 5: >= 70 -> Overbought (Hindari Mengejar Harga / Rawan Profit Taking)
    cond_sell_ob = (prev_rsi < overbought) & (df["RSI"] >= overbought)

    # Aturan 4: Turun ke zona 30 - 45 tanpa pantulan -> Bearish Momentum (Wait & See)
    cond_warn_bear = (prev_rsi >= 45.0) & (df["RSI"] < 45.0) & (df["RSI"] > oversold)

    conditions = [
        cond_buy_reversal,
        cond_buy_bow,
        cond_buy_trend,
        cond_sell_ob,
        cond_warn_bear,
    ]
    choices = [
        "LAYAK BELI: Reversal / Bottom Fishing (> 30) 🟢",
        "LAYAK BELI: Buy on Weakness (45–55 > EMA-20) 🟢",
        "LAYAK BELI: Trend Following (55–65 Cross MA-9) 🟢",
        "HINDARI / PROFIT TAKING: Overbought (≥ 70) 🔴",
        "WAIT & SEE: Bearish Momentum (30–45) 🟡",
    ]
    df["RSI_Signal"] = np.select(conditions, choices, default="Netral")

    # 3. Evaluasi Kondisi & Keputusan Entry pada Bar Terkini
    last_rsi = float(df["RSI"].iloc[-1])
    prev_rsi_val = float(df["RSI"].iloc[-2]) if len(df) >= 2 else last_rsi
    last_rsi_ma = float(df["RSI_MA"].iloc[-1])
    prev_rsi_ma = float(df["RSI_MA"].iloc[-2]) if len(df) >= 2 else last_rsi_ma
    close_now = float(df["Close"].iloc[-1])
    ema20_now = float(df["EMA_20"].iloc[-1])

    if prev_rsi_val <= oversold and last_rsi > oversold:
        active_rule = 1
        market_condition = "Oversold (Jenuh Jual) — Memantul Naik"
        entry_decision = "Layak Beli (Reversal / Bottom Fishing) 🟢"
        action_hint = (
            f"Garis RSI baru saja memotong ke atas melewati angka {oversold} "
            f"({prev_rsi_val:.1f} → {last_rsi:.1f}). Konfirmasi pantulan awal sudah terbentuk."
        )
    elif last_rsi <= oversold:
        active_rule = 1
        market_condition = f"Oversold (Jenuh Jual ≤ {oversold}) — Belum Memantul"
        entry_decision = "Tunggu Pantulan Naik (> 30) 🟡"
        action_hint = (
            f"RSI berada di level {last_rsi:.1f} (≤ {oversold}) namun masih bergerak turun. "
            f"Tunggu garis RSI memotong ke atas melewati angka {oversold}, jangan tangkap pisau jatuh."
        )
    elif last_rsi >= overbought:
        active_rule = 5
        market_condition = f"Overbought (Jenuh Beli ≥ {overbought})"
        entry_decision = "Hindari Mengejar Harga 🔴"
        action_hint = (
            f"RSI sudah mencapai {last_rsi:.1f} (≥ {overbought}). Kenaikan sudah terlalu tinggi "
            "dan rawan aksi ambil untung (profit taking) jangka pendek."
        )
    elif 45.0 <= last_rsi <= 55.0:
        active_rule = 2
        if close_now > ema20_now:
            market_condition = "Koreksi Sehat dalam Uptrend (45–55 > EMA-20)"
            entry_decision = "Layak Beli (Buy on Weakness) 🟢"
            action_hint = (
                f"RSI tertahan di area koreksi sehat ({last_rsi:.1f}) sementara harga masih di atas "
                f"EMA-20 (Rp {close_now:,.0f} > Rp {ema20_now:,.0f}). Ideal untuk masuk sebelum tren lanjut naik."
            )
        else:
            market_condition = "Rentang 45–55 (Harga Masih di Bawah EMA-20)"
            entry_decision = "Tunggu Harga Tembus EMA-20 🟡"
            action_hint = (
                f"RSI berada di {last_rsi:.1f}, tetapi harga (Rp {close_now:,.0f}) belum berada di atas "
                f"EMA-20 (Rp {ema20_now:,.0f}). Tunggu harga menembus EMA-20 untuk konfirmasi uptrend."
            )
    elif 55.0 < last_rsi < overbought:
        active_rule = 3
        is_fresh_cross = (prev_rsi_val <= prev_rsi_ma) and (last_rsi > last_rsi_ma)
        if last_rsi <= 65.0 and is_fresh_cross:
            market_condition = "Bullish Momentum (55–65 Cross ke Atas MA-9)"
            entry_decision = "Layak Beli (Trend Following) 🟢"
            action_hint = (
                f"RSI ({last_rsi:.1f}) memotong ke atas garis sinyal MA-9 ({last_rsi_ma:.1f}) "
                "di rentang 55–65. Tenaga beli dominan dan tren naik terkonfirmasi sehat."
            )
        elif last_rsi <= 65.0 and last_rsi > last_rsi_ma:
            market_condition = "Bullish Momentum (55–65 di Atas MA-9)"
            entry_decision = "Layak Beli / Hold Uptrend (Trend Following) 🟢"
            action_hint = (
                f"RSI ({last_rsi:.1f}) bergerak konsisten di atas garis sinyal MA-9 ({last_rsi_ma:.1f}) "
                "dalam zona bullish sehat (55–65)."
            )
        else:
            market_condition = f"Momentum Kuat ({last_rsi:.1f}) — Menuju Overbought / di Bawah MA-9"
            entry_decision = "Pantau Ketat / Tunggu Cross MA-9 🟡"
            action_hint = (
                f"RSI berada di {last_rsi:.1f} (MA-9: {last_rsi_ma:.1f}). Tunggu RSI memotong "
                "ke atas MA-9 di rentang 55–65 atau hindari jika sudah mendekati 70."
            )
    else:
        active_rule = 4
        market_condition = "Bearish Momentum (30–45 Tanpa Pantulan)"
        entry_decision = "Tunggu Dulu (Wait & See) 🟡"
        action_hint = (
            f"RSI berada di area {last_rsi:.1f} (30–45) tanpa pantulan dari zona oversold. "
            "Tekanan jual masih kuat; tunggu konfirmasi Bullish Divergence atau penembusan ke atas EMA-20."
        )

    # 4. Deteksi RSI Divergence (25 Hari Terakhir)
    divergence_status = "Tidak Terdeteksi Divergensi"
    if len(df) >= 25:
        prior_win = df.iloc[-25:-6]
        recent_win = df.iloc[-6:]

        if (
            recent_win["Low"].min() < prior_win["Low"].min()
            and recent_win["RSI"].min() > (prior_win["RSI"].min() + 2.0)
            and recent_win["RSI"].min() < 45.0
        ):
            divergence_status = "BULLISH DIVERGENCE (Potensi Reversal Naik Kuat) 🟢"
        elif (
            recent_win["High"].max() > prior_win["High"].max()
            and recent_win["RSI"].max() < (prior_win["RSI"].max() - 2.0)
            and recent_win["RSI"].max() > 55.0
        ):
            divergence_status = "BEARISH DIVERGENCE (Waspada Pelemahan Tren) 🔴"

    # 5. Backtest Akurasi 3 Sinyal Layak Beli RSI (Hold 10 Hari Bursa)
    bt_df = df.copy()
    bt_df["Fwd_Ret_10D"] = (bt_df["Close"].shift(-10) / bt_df["Close"]) - 1.0
    buy_trades = bt_df[bt_df["RSI_Signal"].str.startswith("LAYAK BELI")].dropna(subset=["Fwd_Ret_10D"])

    total_buy_signals = len(buy_trades)
    if total_buy_signals > 0:
        win_rate = float((buy_trades["Fwd_Ret_10D"] > 0).mean() * 100.0)
        avg_return = float(buy_trades["Fwd_Ret_10D"].mean() * 100.0)
    else:
        win_rate = 0.0
        avg_return = 0.0

    # 6. Tabel Matriks Panduan 5 Zona RSI + Penanda Posisi Saham Saat Ini
    guide_table = [
        {
            "Status Saham Kini": "👈 POSISI SAAT INI" if active_rule == 1 else "-",
            "Rentang Nilai RSI": f"≤ {oversold} lalu memantul naik (> {oversold})",
            "Kondisi Pasar": "Oversold (Jenuh Jual)",
            "Keputusan & Titik Masuk (Entry) Ideal": f"Layak Beli (Reversal / Bottom Fishing) 🟢 — Tunggu garis RSI memotong ke atas melewati angka {oversold}, bukan saat masih menukik turun di bawah {oversold}.",
        },
        {
            "Status Saham Kini": "👈 POSISI SAAT INI" if active_rule == 2 else "-",
            "Rentang Nilai RSI": "45 – 55 (saat harga di atas EMA-20)",
            "Kondisi Pasar": "Koreksi Sehat dalam Uptrend",
            "Keputusan & Titik Masuk (Entry) Ideal": "Layak Beli (Buy on Weakness) 🟢 — Pada saham yang sedang tren naik kuat, koreksi biasanya tertahan di area RSI 45–50 sebelum lanjut naik.",
        },
        {
            "Status Saham Kini": "👈 POSISI SAAT INI" if active_rule == 3 else "-",
            "Rentang Nilai RSI": "55 – 65 (RSI memotong ke atas MA-9)",
            "Kondisi Pasar": "Bullish Momentum",
            "Keputusan & Titik Masuk (Entry) Ideal": "Layak Beli (Trend Following) 🟢 — Tenaga beli sedang dominan dan tren naik terkonfirmasi sehat.",
        },
        {
            "Status Saham Kini": "👈 POSISI SAAT INI" if active_rule == 4 else "-",
            "Rentang Nilai RSI": "30 – 45 (tanpa pantulan)",
            "Kondisi Pasar": "Bearish Momentum",
            "Keputusan & Titik Masuk (Entry) Ideal": "Tunggu Dulu (Wait & See) 🟡 — Tekanan jual masih kuat; tunggu konfirmasi Bullish Divergence atau penembusan ke atas EMA-20.",
        },
        {
            "Status Saham Kini": "👈 POSISI SAAT INI" if active_rule == 5 else "-",
            "Rentang Nilai RSI": f"≥ {overbought}",
            "Kondisi Pasar": "Overbought (Jenuh Beli)",
            "Keputusan & Titik Masuk (Entry) Ideal": "Hindari Mengejar Harga 🔴 — Kenaikan sudah terlalu tinggi dan rawan aksi ambil untung (profit taking) jangka pendek.",
        },
    ]

    return {
        "df": df,
        "last_rsi": last_rsi,
        "last_rsi_ma": last_rsi_ma,
        "rsi_delta": last_rsi - prev_rsi_val,
        "active_rule": active_rule,
        "market_condition": market_condition,
        "entry_decision": entry_decision,
        "action_hint": action_hint,
        "divergence_status": divergence_status,
        "last_signal": str(df["RSI_Signal"].iloc[-1]),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_buy_signals": total_buy_signals,
        "guide_table": guide_table,
    }
