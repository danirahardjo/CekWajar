import numpy as np
import pandas as pd


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
