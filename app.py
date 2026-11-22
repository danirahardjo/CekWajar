import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="CekWajar | Valuasi & Prediksi Saham IDX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# KAMUS SENTIMEN FINANSIAL (ID & EN)
# ==========================================
BULLISH_WORDS = {
    # Bahasa Indonesia
    "laba naik": 2.0, "untung": 1.5, "rekor": 1.8, "dividen": 1.7, "ekspansi": 1.5,
    "tumbuh": 1.4, "pertumbuhan": 1.4, "menguat": 1.5, "melonjak": 1.8, "akuisisi": 1.3,
    "buyback": 1.6, "surplus": 1.4, "positif": 1.2, "optimis": 1.3, "rekomendasi beli": 1.8,
    "target harga naik": 1.8, "kinerja cemerlang": 1.9, "pendapatan naik": 1.6, "prospek cerah": 1.5,
    # Bahasa Inggris (Berita Global)
    "surge": 1.8, "soar": 1.8, "profit": 1.5, "growth": 1.4, "dividend": 1.6,
    "beat": 1.6, "outperform": 1.7, "buy": 1.5, "upgrade": 1.8, "bullish": 1.7,
    "expansion": 1.5, "record high": 1.9, "rally": 1.6, "strong": 1.3, "gain": 1.3,
    "rate cut": 1.5, "inflow": 1.5, "recovery": 1.4, "resilient": 1.3,
}

BEARISH_WORDS = {
    # Bahasa Indonesia
    "rugi": -2.0, "kerugian": -2.0, "anjlok": -1.9, "melemah": -1.4, "turun": -1.2,
    "penurunan": -1.3, "utang": -1.2, "gagal bayar": -2.2, "pailit": -2.5, "korupsi": -2.2,
    "sanksi": -1.7, "suspensi": -2.0, "inflasi": -1.2, "suku bunga naik": -1.5, "negatif": -1.3,
    "koreksi": -1.2, "jual bersih": -1.5, "net sell": -1.5, "pangkas": -1.4, "lesu": -1.4,
    # Bahasa Inggris (Berita Global)
    "loss": -1.8, "drop": -1.5, "plunge": -1.9, "fall": -1.3, "decline": -1.4,
    "downgrade": -1.8, "sell": -1.5, "bearish": -1.7, "debt": -1.2, "default": -2.2,
    "lawsuit": -1.7, "miss": -1.6, "weak": -1.4, "recession": -1.8, "inflation": -1.2,
    "rate hike": -1.5, "outflow": -1.5, "slump": -1.7, "risk": -1.1, "warning": -1.5,
}


def analyze_headline_sentiment(text: str) -> float:
    """Menghitung skor sentimen judul berita pada rentang -1.0 (Bearish) hingga +1.0 (Bullish)."""
    if not text:
        return 0.0
    lower_text = text.lower()
    score = 0.0
    for phrase, weight in BULLISH_WORDS.items():
        if phrase in lower_text:
            score += weight
    for phrase, weight in BEARISH_WORDS.items():
        if phrase in lower_text:
            score += weight
    return float(np.tanh(score / 2.5))


# ==========================================
# FUNGSI PENGAMBILAN DATA SAHAM & FUNDAMENTAL
# ==========================================
@st.cache_data(ttl=1800)
def fetch_stock_data(ticker_symbol: str, period: str = "2y"):
    """Mengambil data historis harga & fundamental dari Yahoo Finance untuk kode saham IDX."""
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


# ==========================================
# FUNGSI PENGAMBILAN BERITA (DALAM & LUAR NEGERI)
# ==========================================
def fetch_google_news_rss(query: str, region: str = "ID", max_items: int = 10) -> list[dict]:
    """Mengambil berita terkini dari Google News RSS tanpa memerlukan API Key."""
    encoded_q = urllib.parse.quote(query)
    if region == "ID":
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=id&gl=ID&ceid=ID:id"
    else:
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"

    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for item in root.findall("./channel/item")[:max_items]:
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            pub_date = item.findtext("pubDate", default="")
            source = item.findtext("source", default="Google News")
            sentiment = analyze_headline_sentiment(title)
            articles.append({
                "Judul": title,
                "Sumber": source,
                "Tanggal": pub_date[:16] if pub_date else "-",
                "Skor Sentimen": round(sentiment, 3),
                "Label": "Bullish 🟢" if sentiment > 0.15 else ("Bearish 🔴" if sentiment < -0.15 else "Netral ⚪"),
                "Link": link,
            })
    except Exception:
        pass
    return articles


@st.cache_data(ttl=1800)
def get_dual_market_news(base_code: str, company_name: str):
    """Mengambil berita dalam negeri (Indonesia) dan luar negeri (Global)."""
    short_name = company_name.replace("PT ", "").replace(" Tbk", "").strip() if company_name else base_code
    query_id = f'"{base_code}" OR "{short_name}" saham'
    query_global = f'"{short_name}" OR "{base_code}" Indonesia stock OR economy'

    news_id = fetch_google_news_rss(query_id, region="ID", max_items=10)
    news_global = fetch_google_news_rss(query_global, region="US", max_items=10)

    df_id = pd.DataFrame(news_id)
    df_global = pd.DataFrame(news_global)

    score_id = float(df_id["Skor Sentimen"].mean()) if not df_id.empty else 0.0
    score_global = float(df_global["Skor Sentimen"].mean()) if not df_global.empty else 0.0
    return df_id, df_global, score_id, score_global


# ==========================================
# MODUL VALUASI HARGA WAJAR SAHAM
# ==========================================
def calculate_fair_value(
    current_price: float,
    eps: float,
    bvps: float,
    fair_per: float,
    fair_pbv: float,
) -> dict:
    """Menghitung harga wajar menggunakan Graham Number, Relative PER, dan Relative PBV."""
    models = {}

    # 1. Benjamin Graham Number: sqrt(22.5 * EPS * BVPS)
    if eps > 0 and bvps > 0:
        graham = math.sqrt(22.5 * eps * bvps)
        models["Graham Number"] = graham

    # 2. Relative PER Valuation
    if eps > 0:
        models[f"Valuasi PER ({fair_per:.1f}x)"] = eps * fair_per

    # 3. Relative PBV Valuation
    if bvps > 0:
        models[f"Valuasi PBV ({fair_pbv:.1f}x)"] = bvps * fair_pbv

    if models:
        composite_fair_value = float(np.mean(list(models.values())))
        mos_pct = ((composite_fair_value - current_price) / composite_fair_value) * 100
    else:
        composite_fair_value = 0.0
        mos_pct = 0.0

    if composite_fair_value == 0:
        status = "Data Fundamental Tidak Mencukupi (EPS/BVPS Negatif)"
    elif mos_pct >= 15:
        status = "UNDERVALUED (Murah / Layak Koleksi) 🟢"
    elif mos_pct <= -15:
        status = "OVERVALUED (Mahal) 🔴"
    else:
        status = "FAIRLY VALUED (Harga Wajar) 🟡"

    return {
        "models": models,
        "composite": composite_fair_value,
        "mos_pct": mos_pct,
        "status": status,
    }


# ==========================================
# MODUL PREDIKSI HARGA (HISTORIS + SENTIMEN)
# ==========================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membentuk fitur teknikal dan return historis untuk pelatihan ML."""
    data = df[["Close", "Volume"]].copy()
    data["Return_1D"] = data["Close"].pct_change()
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["MA_Ratio"] = data["MA_5"] / data["MA_20"]
    data["Volatility_10"] = data["Return_1D"].rolling(10).std()
    data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1

    # RSI 14 hari
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    data["RSI_14"] = data["RSI_14"].fillna(50.0)

    # Target: Harga penutupan hari berikutnya
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

    # Latih ulang pada 100% data historis untuk proyeksi ke depan
    model.fit(X, y)

    # Iterasi prediksi N hari bursa ke depan
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

        # Efek peluruhan sentimen berita harian
        decay = math.exp(-0.08 * step)
        sentiment_drift = composite_sentiment * sentiment_weight * 0.004 * decay
        adjusted_pred = base_pred * (1 + sentiment_drift)

        # Rentang ketidakpastian (confidence interval)
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


# ==========================================
# ANTARMUKA PENGGUNA (STREAMLIT UI)
# ==========================================
st.title("📈 CekWajar")
st.caption("Cek Harga Wajar & Prediksi Saham Indonesia Berbasis Berita Domestik, Global, dan Historis")

with st.sidebar:
    st.header("⚙️ Parameter Analisis")
    preset_tickers = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "ADRO", "ICBP", "GOTO", "AMMN", "Custom..."]
    selected_preset = st.selectbox("Pilih Emiten Populer IDX", preset_tickers, index=0)

    if selected_preset == "Custom...":
        ticker_input = st.text_input("Masukkan Kode Saham BEI", value="BBNI").upper()
    else:
        ticker_input = selected_preset

    period_opt = st.selectbox("Rentang Data Historis", ["1y", "2y", "3y", "5y"], index=1)
    forecast_days = st.slider("Horizon Prediksi (Hari Bursa)", min_value=5, max_value=30, value=14)

    st.divider()
    st.subheader("📐 Asumsi Valuasi Wajar")
    fair_per = st.slider("Target PER Wajar (x)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
    fair_pbv = st.slider("Target PBV Wajar (x)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)

    st.divider()
    st.subheader("📰 Bobot Sentimen Berita")
    weight_id = st.slider("Bobot Berita Dalam Negeri (%)", 0, 100, 60)
    weight_global = 100 - weight_id
    st.caption(f"Bobot Berita Luar Negeri otomatis: **{weight_global}%**")
    sentiment_impact = st.slider("Sensitivitas Sentimen ke Prediksi", 0.0, 2.0, 1.0, 0.1)

# Eksekusi Pengambilan Data
with st.spinner(f"Mengambil data pasar & berita untuk {ticker_input}..."):
    clean_ticker, hist_df, info = fetch_stock_data(ticker_input, period=period_opt)

if hist_df.empty:
    st.error(f"Data untuk kode saham **{clean_ticker}** tidak ditemukan. Pastikan kode emiten BEI valid (contoh: BBCA, TLKM, ASII).")
    st.stop()

company_name = info.get("longName") or info.get("shortName") or clean_ticker
sector = info.get("sector", "Tidak Tersedia")
current_price = float(info.get("currentPrice") or hist_df["Close"].iloc[-1])

# Ambil fundamental dengan fallback input manual jika data Yahoo Finance kosong
raw_eps = info.get("trailingEps")
raw_bvps = info.get("bookValue")

with st.sidebar:
    st.divider()
    st.subheader("🧾 Data Fundamental (Opsional Override)")
    eps_val = st.number_input("EPS (Laba per Saham - Rp)", value=float(raw_eps) if raw_eps else 0.0, step=10.0)
    bvps_val = st.number_input("BVPS (Nilai Buku per Saham - Rp)", value=float(raw_bvps) if raw_bvps else 0.0, step=50.0)

# Hitung Valuasi Harga Wajar
valuation = calculate_fair_value(current_price, eps_val, bvps_val, fair_per, fair_pbv)

# Ambil & Analisis Sentimen Berita Dalam Negeri & Luar Negeri
base_code = clean_ticker.replace(".JK", "")
df_news_id, df_news_global, score_id, score_global = get_dual_market_news(base_code, company_name)
composite_sentiment = (score_id * (weight_id / 100.0)) + (score_global * (weight_global / 100.0))

# Jalankan Prediksi Harga Saham
forecast_df, mae, mape = train_and_forecast(
    hist_df=hist_df,
    forecast_days=forecast_days,
    composite_sentiment=composite_sentiment,
    sentiment_weight=sentiment_impact,
)
target_price_end = float(forecast_df["Prediksi_Harga"].iloc[-1])
expected_change_pct = ((target_price_end - current_price) / current_price) * 100

# ==========================================
# RINGKASAN EKSEKUTIF (KPI METRICS)
# ==========================================
st.subheader(f"🏢 {company_name} (`{clean_ticker}`)")
st.markdown(f"**Sektor:** {sector}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Harga Pasar Saat Ini", f"Rp {current_price:,.0f}")
with col2:
    fv = valuation["composite"]
    st.metric(
        "Estimasi Harga Wajar",
        f"Rp {fv:,.0f}" if fv > 0 else "N/A",
        delta=f"MoS {valuation['mos_pct']:+.1f}%" if fv > 0 else None,
    )
with col3:
    st.metric(
        f"Prediksi ({forecast_days} Hari Bursa)",
        f"Rp {target_price_end:,.0f}",
        delta=f"{expected_change_pct:+.2f}%",
    )
with col4:
    sent_label = "Bullish 🟢" if composite_sentiment > 0.1 else ("Bearish 🔴" if composite_sentiment < -0.1 else "Netral ⚪")
    st.metric("Skor Sentimen Gabungan", f"{composite_sentiment:+.2f} ({sent_label})")

st.info(f"**Status Valuasi Fundamental:** {valuation['status']}")

# ==========================================
# TABS DETAIL ANALISIS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📊 Grafik Prediksi & Harga Wajar",
    "💰 Rincian Valuasi Fundamental",
    "📰 Analisis Berita Dalam & Luar Negeri",
])

with tab1:
    st.markdown("### Proyeksi Harga Saham (Historis + Sentimen Berita)")
    recent_hist = hist_df.iloc[-120:].copy()
    recent_hist["MA20"] = recent_hist["Close"].rolling(20).mean()

    fig = go.Figure()

    # Candlestick historis
    fig.add_trace(go.Candlestick(
        x=recent_hist.index,
        open=recent_hist["Open"],
        high=recent_hist["High"],
        low=recent_hist["Low"],
        close=recent_hist["Close"],
        name="Harga Historis",
    ))

    # Moving Average 20
    fig.add_trace(go.Scatter(
        x=recent_hist.index,
        y=recent_hist["MA20"],
        mode="lines",
        name="MA 20 Hari",
        line=dict(color="orange", width=1.5),
    ))

    # Garis Prediksi CekWajar
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Prediksi_Harga"],
        mode="lines+markers",
        name="Prediksi CekWajar",
        line=dict(color="#00CC96", width=2.5, dash="dash"),
    ))

    # Pita Rentang Estimasi (Upper & Lower)
    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["Batas_Atas"]) + list(forecast_df["Batas_Bawah"][::-1]),
        fill="toself",
        fillcolor="rgba(0, 204, 150, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Rentang Volatilitas Prediksi",
    ))

    # Garis Horizontal Harga Wajar
    if valuation["composite"] > 0:
        fig.add_hline(
            y=valuation["composite"],
            line_dash="dot",
            line_color="cyan",
            annotation_text=f"Harga Wajar: Rp {valuation['composite']:,.0f}",
            annotation_position="top left",
        )

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        yaxis_title="Harga Saham (IDR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Akurasi Uji Historis (MAPE)", f"{mape:.2f}%")
    c_m2.metric("Rata-rata Error (MAE)", f"Rp {mae:,.0f}")
    c_m3.metric("Sentimen Domestik vs Global", f"ID: {score_id:+.2f} | Global: {score_global:+.2f}")

    with st.expander("Lihat Tabel Proyeksi Harian"):
        disp_forecast = forecast_df.copy()
        disp_forecast.index = disp_forecast.index.strftime("%Y-%m-%d")
        st.dataframe(disp_forecast.style.format("Rp {:,.0f}"), use_container_width=True)

with tab2:
    st.markdown("### Perbandingan Metode Valuasi Intrinsik & Relatif")
    if valuation["models"]:
        val_rows = []
        for method_name, fair_val in valuation["models"].items():
            diff_pct = ((fair_val - current_price) / current_price) * 100
            val_rows.append({
                "Metode Valuasi": method_name,
                "Harga Wajar (Rp)": fair_val,
                "Harga Pasar (Rp)": current_price,
                "Potensi Upside/Downside (%)": diff_pct,
            })
        df_val = pd.DataFrame(val_rows)
        st.dataframe(
            df_val.style.format({
                "Harga Wajar (Rp)": "Rp {:,.0f}",
                "Harga Pasar (Rp)": "Rp {:,.0f}",
                "Potensi Upside/Downside (%)": "{:+.2f}%",
            }),
            use_container_width=True,
        )
    else:
        st.warning("EPS atau BVPS bernilai nol/negatif. Silakan masukkan estimasi EPS/BVPS secara manual di sidebar kiri.")

    st.markdown("#### Metrik Kunci Emiten")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    f_col1.metric("Trailing PER", f"{info.get('trailingPE', 0):.2f}x" if info.get("trailingPE") else "-")
    f_col2.metric("Price to Book (PBV)", f"{info.get('priceToBook', 0):.2f}x" if info.get("priceToBook") else "-")
    f_col3.metric("Return on Equity (ROE)", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get("returnOnEquity") else "-")
    div_yield = info.get("dividendYield")
    f_col4.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield and div_yield < 1 else (f"{div_yield:.2f}%" if div_yield else "-"))

with tab3:
    col_id, col_gl = st.columns(2)
    with col_id:
        st.markdown(f"### 🇮🇩 Berita Dalam Negeri (Skor: `{score_id:+.2f}`)")
        if not df_news_id.empty:
            st.dataframe(
                df_news_id[["Judul", "Sumber", "Skor Sentimen", "Label", "Link"]],
                column_config={"Link": st.column_config.LinkColumn("Baca")},
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Tidak ditemukan berita domestik terbaru untuk kata kunci ini.")

    with col_gl:
        st.markdown(f"### 🌍 Berita Luar Negeri / Global (Skor: `{score_global:+.2f}`)")
        if not df_news_global.empty:
            st.dataframe(
                df_news_global[["Judul", "Sumber", "Skor Sentimen", "Label", "Link"]],
                column_config={"Link": st.column_config.LinkColumn("Read")},
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Tidak ditemukan berita internasional terbaru untuk emiten ini.")

st.divider()
st.caption("⚠️ **Disclaimer:** CekWajar adalah alat bantu riset kuantitatif. Hasil valuasi dan prediksi Machine Learning bukan merupakan ajakan membeli atau menjual saham (DYOR).")