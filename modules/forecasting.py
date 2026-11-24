import math
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membentuk fitur teknikal dan return historis untuk pelatihan ML."""
    data = df[["Close", "Volume"]].copy()
    data["Return_1D"] = data["Close"].pct_change()
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["MA_Ratio"] = data["MA_5"] / data["MA_20"]
    data["Volatility_10"] = data["Return_1D"].rolling(10).std()
    data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    data["RSI_14"] = data["RSI_14"].fillna(50.0)

    data["Target_Next_Close"] = data["Close"].shift(-1)
    return data.dropna()


def train_and_forecast(
    hist_df: pd.DataFrame,
    forecast_days: int,
    composite_sentiment: float,
    sentiment_weight: float,
):
    """Melatih Random Forest Regressor & memprediksi harga masa depan dengan pengaruh sentimen."""
    featured = build_features(hist_df)
    feature_cols = ["Close", "MA_5", "MA_20", "MA_Ratio", "Volatility_10", "Momentum_5", "RSI_14"]

    X = featured[feature_cols]
    y = featured["Target_Next_Close"]

    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    model.fit(X, y)

    rolling_closes = list(hist_df["Close"].iloc[-30:].values)
    last_date = hist_df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

    predictions = []
    upper_band = []
    lower_band = []
    hist_daily_vol = float(hist_df["Close"].pct_change().std())

    for step in range(forecast_days):
        s_close = pd.Series(rolling_closes)
        c_val = s_close.iloc[-1]
        ma5 = s_close.iloc[-5:].mean()
        ma20 = s_close.iloc[-20:].mean()
        ma_ratio = ma5 / ma20
        vol10 = s_close.pct_change().iloc[-10:].std()
        mom5 = (c_val / s_close.iloc[-6]) - 1

        delta = s_close.diff()
        gain = delta.clip(lower=0).iloc[-14:].mean()
        loss = (-delta.clip(upper=0)).iloc[-14:].mean()
        rsi14 = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 50.0

        row = pd.DataFrame(
            [[c_val, ma5, ma20, ma_ratio, vol10, mom5, rsi14]],
            columns=feature_cols,
        )
        base_pred = float(model.predict(row)[0])

        decay = math.exp(-0.08 * step)
        sentiment_drift = composite_sentiment * sentiment_weight * 0.004 * decay
        adjusted_pred = base_pred * (1 + sentiment_drift)

        spread = adjusted_pred * hist_daily_vol * math.sqrt(step + 1) * 1.25
        predictions.append(adjusted_pred)
        upper_band.append(adjusted_pred + spread)
        lower_band.append(max(1.0, adjusted_pred - spread))

        rolling_closes.append(adjusted_pred)

    forecast_df = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi_Harga": predictions,
        "Batas_Atas": upper_band,
        "Batas_Bawah": lower_band,
    }).set_index("Tanggal")

    return forecast_df, mae, mape
