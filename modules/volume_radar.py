import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

IDX_WATCHLISTS = {
    "Semua Saham Likuid IDX (60+ Emiten)": [
        "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "BBTN", "NISP", "BNGA",
        "TLKM", "ISAT", "EXCL", "MTEL", "TOWR",
        "ASII", "AUTO", "UNTR",
        "ADRO", "ITMG", "PTBA", "PGAS", "MEDC", "AKRA", "ELSA", "RAJA",
        "AMMN", "ANTM", "INCO", "MDKA", "MBMA", "NCKL", "TINS", "PSAB",
        "BRPT", "TPIA", "CUAN", "BREN", "PGEO",
        "ICBP", "INDF", "MYOR", "UNVR", "KLBF", "SIDO", "MIKA", "HEAL", "CPIN", "JPFA",
        "AMRT", "ACES", "MAPI", "ERAA",
        "GOTO", "BUKA", "EMTK", "SCMA", "WIFI",
        "BSDE", "CTRA", "PWON", "SMRA", "PANI",
        "JSMR", "SMGR", "INTP", "INKP", "TKIM",
        "BULL", "SMDR", "BIRD", "ASSA",
    ],
    "Perbankan & Keuangan": [
        "BBCA", "BBRI", "BMRI", "BBNI", "BRIS", "BBTN", "NISP", "BNGA", "BDMN", "BFIN", "ARTO",
    ],
    "Energi, Tambang & Komoditas": [
        "ADRO", "ITMG", "PTBA", "PGAS", "MEDC", "AKRA", "ELSA", "RAJA",
        "AMMN", "ANTM", "INCO", "MDKA", "MBMA", "NCKL", "TINS", "PSAB", "BUMI", "DEWA",
    ],
    "Konsumsi, Ritel & Kesehatan": [
        "ICBP", "INDF", "MYOR", "UNVR", "KLBF", "SIDO", "MIKA", "HEAL",
        "CPIN", "JPFA", "AMRT", "ACES", "MAPI", "ERAA",
    ],
    "Teknologi, Telko & Media": [
        "TLKM", "ISAT", "EXCL", "MTEL", "TOWR", "GOTO", "BUKA", "EMTK", "SCMA", "WIFI",
    ],
    "Properti, Infrastruktur & Pelayaran": [
        "BSDE", "CTRA", "PWON", "SMRA", "PANI", "JSMR", "SMGR", "INTP",
        "INKP", "TKIM", "BULL", "SMDR", "BIRD", "ASSA",
    ],
}


def analyze_massive_volume(
    intraday_df: pd.DataFrame,
    rvol_threshold: float = 2.5,
    min_value_miliar: float = 0.5,
) -> pd.DataFrame:
    """Menganalisis lonjakan volume (RVOL & Z-Score) serta memisahkan tekanan Beli (HAKA) vs Jual (HAKI)."""
    df = intraday_df.copy()

    df["Vol_MA20"] = df["Volume"].rolling(20, min_periods=5).mean()
    df["Vol_Std20"] = df["Volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    df["RVOL"] = (df["Volume"] / df["Vol_MA20"]).fillna(1.0)
    df["Vol_ZScore"] = ((df["Volume"] - df["Vol_MA20"]) / df["Vol_Std20"]).fillna(0.0)

    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["Est_Value_Rp"] = df["Volume"] * df["Typical_Price"]
    df["Date_Only"] = df.index.date
    df["TP_Vol"] = df["Typical_Price"] * df["Volume"]
    df["VWAP"] = (
        df.groupby("Date_Only")["TP_Vol"].cumsum()
        / df.groupby("Date_Only")["Volume"].cumsum()
    )

    bar_range = df["High"] - df["Low"]
    prev_close = df["Close"].shift(1).fillna(df["Open"])
    fallback_ratio = np.where(
        df["Close"] > prev_close, 0.75, np.where(df["Close"] < prev_close, 0.25, 0.50)
    )
    df["Buy_Ratio"] = np.where(
        bar_range > 0,
        (df["Close"] - df["Low"]) / bar_range,
        fallback_ratio,
    )
    df["Buy_Ratio"] = df["Buy_Ratio"].clip(0.0, 1.0)

    df["Est_Buy_Vol"] = df["Volume"] * df["Buy_Ratio"]
    df["Est_Sell_Vol"] = df["Volume"] * (1.0 - df["Buy_Ratio"])
    df["Net_Vol_Delta"] = df["Est_Buy_Vol"] - df["Est_Sell_Vol"]
    df["CVD_Harian"] = df.groupby("Date_Only")["Net_Vol_Delta"].cumsum()

    min_value_rp = min_value_miliar * 1_000_000_000
    is_spike = (df["RVOL"] >= rvol_threshold) & (df["Est_Value_Rp"] >= min_value_rp)

    conditions = [
        is_spike & (df["Buy_Ratio"] >= 0.60),
        is_spike & (df["Buy_Ratio"] <= 0.40),
        is_spike,
    ]
    choices = [
        "MASIF BUY / AKUMULASI (HAKA) 🟢",
        "MASIF SELL / DISTRIBUSI (HAKI) 🔴",
        "MASIF CROSSING / TARIK-MENARIK 🟡",
    ]
    df["Sinyal_Volume"] = np.select(conditions, choices, default="Normal")
    df["Is_Spike"] = is_spike

    return df


@st.cache_data(ttl=60)
def scan_market_volume_screener(
    tickers_tuple: tuple[str, ...],
    interval: str = "5m",
    rvol_threshold: float = 2.5,
    min_value_miliar: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Memindai banyak emiten IDX sekaligus untuk mendeteksi lonjakan volume jual/beli masif."""
    clean_tickers = []
    for t in tickers_tuple:
        code = t.strip().upper()
        if code:
            clean_tickers.append(code if code.endswith(".JK") else f"{code}.JK")
    clean_tickers = list(dict.fromkeys(clean_tickers))

    if not clean_tickers:
        return pd.DataFrame(), {}

    period_map = {"1m": "1d", "5m": "5d", "15m": "5d"}
    period = period_map.get(interval, "5d")

    try:
        raw = yf.download(
            tickers=clean_tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame(), {}

    if raw is None or raw.empty:
        return pd.DataFrame(), {}

    summary_rows = []
    analyzed_map: dict[str, pd.DataFrame] = {}

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    level0_vals = set(raw.columns.get_level_values(0)) if is_multi else set()

    for jk_ticker in clean_tickers:
        try:
            if is_multi:
                if "Close" in level0_vals:
                    df_t = raw.xs(jk_ticker, level=1, axis=1).copy()
                elif jk_ticker in level0_vals:
                    df_t = raw[jk_ticker].copy()
                else:
                    continue
            else:
                df_t = raw.copy()

            if "Close" not in df_t.columns or "Volume" not in df_t.columns:
                continue

            df_t = df_t.dropna(subset=["Close", "Volume"]).copy()
            df_t = df_t[df_t["Volume"] > 0].copy()
            if len(df_t) < 6:
                continue

            df_t.index = pd.to_datetime(df_t.index)
            if df_t.index.tz is None:
                df_t.index = df_t.index.tz_localize("UTC").tz_convert("Asia/Jakarta")
            else:
                df_t.index = df_t.index.tz_convert("Asia/Jakarta")

            vol_df = analyze_massive_volume(
                df_t,
                rvol_threshold=rvol_threshold,
                min_value_miliar=min_value_miliar,
            )

            latest_date = vol_df["Date_Only"].iloc[-1]
            today_df = vol_df[vol_df["Date_Only"] == latest_date]
            prev_df = vol_df[vol_df["Date_Only"] < latest_date]

            prev_close = (
                float(prev_df["Close"].iloc[-1])
                if not prev_df.empty
                else float(today_df["Open"].iloc[0])
            )
            last_bar = today_df.iloc[-1]
            last_price = float(last_bar["Close"])
            chg_pct = ((last_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            spikes_today = today_df[today_df["Is_Spike"]]
            spike_count = len(spikes_today)

            if bool(last_bar["Is_Spike"]):
                status_sinyal = str(last_bar["Sinyal_Volume"])
                prioritas = 3
            elif spike_count > 0:
                last_spike = spikes_today.iloc[-1]
                jam_spike = spikes_today.index[-1].strftime("%H:%M")
                status_sinyal = f"Riwayat {last_spike['Sinyal_Volume']} ({jam_spike})"
                prioritas = 2
            else:
                status_sinyal = "Normal"
                prioritas = 1

            net_flow_miliar = float((today_df["Net_Vol_Delta"] * today_df["Typical_Price"]).sum()) / 1e9
            base_code = jk_ticker.replace(".JK", "")
            analyzed_map[base_code] = vol_df

            summary_rows.append({
                "Kode": base_code,
                "Harga (Rp)": last_price,
                "Perubahan (%)": chg_pct,
                "RVOL": float(last_bar["RVOL"]),
                "Z-Score": float(last_bar["Vol_ZScore"]),
                "Est. Nilai Bar (Rp M)": float(last_bar["Est_Value_Rp"]) / 1e9,
                "Dominasi Beli (%)": float(last_bar["Buy_Ratio"]) * 100,
                "Net Flow Hari Ini (Rp M)": net_flow_miliar,
                "Spike Hari Ini": spike_count,
                "Waktu (WIB)": today_df.index[-1].strftime("%d %b %H:%M"),
                "Sinyal Terdeteksi": status_sinyal,
                "_prioritas": prioritas,
            })
        except Exception:
            continue

    if not summary_rows:
        return pd.DataFrame(), {}

    df_res = pd.DataFrame(summary_rows)
    df_res = (
        df_res.sort_values(by=["_prioritas", "RVOL", "Est. Nilai Bar (Rp M)"], ascending=[False, False, False])
        .drop(columns=["_prioritas"])
        .reset_index(drop=True)
    )
    return df_res, analyzed_map
