import pandas as pd
import streamlit as st

from modules import (
    analyze_massive_volume,
    calculate_fair_value,
    create_forecast_chart,
    create_volume_radar_chart,
    fetch_intraday_volume_data,
    fetch_stock_data,
    get_dual_market_news,
    train_and_forecast,
)

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="CekWajar | Valuasi, Prediksi & Radar Volume Saham IDX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 CekWajar")
st.caption("Valuasi Harga Wajar, Prediksi AI & Radar Volume Jual/Beli Masif Saham Indonesia (IDX)")

# ==========================================
# SIDEBAR PARAMETER ANALISIS
# ==========================================
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
    st.subheader("⚡ Parameter Radar Volume Masif")
    intraday_interval = st.selectbox("Interval Candle Intraday", ["5m", "15m", "1m"], index=0)
    rvol_threshold = st.slider("Ambang Lonjakan Volume (RVOL)", min_value=1.5, max_value=5.0, value=2.5, step=0.1)
    min_val_miliar = st.number_input("Minimal Nilai Transaksi/Bar (Rp Miliar)", min_value=0.0, value=0.5, step=0.5)

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

# ==========================================
# EKSEKUSI PENGAMBILAN DATA & KALKULASI
# ==========================================
with st.spinner(f"Mengambil data pasar & berita untuk {ticker_input}..."):
    clean_ticker, hist_df, info = fetch_stock_data(ticker_input, period=period_opt)

if hist_df.empty:
    st.error(f"Data untuk kode saham **{clean_ticker}** tidak ditemukan. Pastikan kode emiten BEI valid (contoh: BBCA, TLKM, ASII).")
    st.stop()

company_name = info.get("longName") or info.get("shortName") or clean_ticker
sector = info.get("sector", "Tidak Tersedia")
current_price = float(info.get("currentPrice") or hist_df["Close"].iloc[-1])

raw_eps = info.get("trailingEps")
raw_bvps = info.get("bookValue")

with st.sidebar:
    st.divider()
    st.subheader("🧾 Data Fundamental (Opsional Override)")
    eps_val = st.number_input("EPS (Laba per Saham - Rp)", value=float(raw_eps) if raw_eps else 0.0, step=10.0)
    bvps_val = st.number_input("BVPS (Nilai Buku per Saham - Rp)", value=float(raw_bvps) if raw_bvps else 0.0, step=50.0)

valuation = calculate_fair_value(current_price, eps_val, bvps_val, fair_per, fair_pbv)

base_code = clean_ticker.replace(".JK", "")
df_news_id, df_news_global, score_id, score_global = get_dual_market_news(base_code, company_name)
composite_sentiment = (score_id * (weight_id / 100.0)) + (score_global * (weight_global / 100.0))

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
    mos_val = valuation["mos_pct"]
    st.metric(
        "Estimasi Harga Wajar",
        f"Rp {fv:,.0f}" if fv > 0 else "N/A",
        delta=f"{mos_val:+.1f}% (MoS)" if fv > 0 else None,
        delta_color="normal",
    )
with col3:
    st.metric(
        f"Prediksi ({forecast_days} Hari Bursa)",
        f"Rp {target_price_end:,.0f}",
        delta=f"{expected_change_pct:+.2f}%",
    )
with col4:
    sent_analisis = "Bullish 🟢" if composite_sentiment > 0.1 else ("Bearish 🔴" if composite_sentiment < -0.1 else "Netral ⚪")
    st.metric("Skor Sentimen Gabungan", f"{composite_sentiment:+.2f} ({sent_analisis})")

st.info(f"**Status Valuasi Fundamental:** {valuation['status']}")

# ==========================================
# TABS DETAIL ANALISIS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Deteksi Volume Masif (Real-Time)",
    "📊 Grafik Prediksi & Harga Wajar",
    "💰 Rincian Valuasi Fundamental",
    "📰 Analisis Berita Dalam & Luar Negeri",
])

with tab1:
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown(f"### Radar Lonjakan Volume Jual / Beli Intraday (`{intraday_interval}`)")
    with head_col2:
        if st.button("🔄 Refresh Live", use_container_width=True):
            fetch_intraday_volume_data.clear()
            st.rerun()

    raw_intraday = fetch_intraday_volume_data(clean_ticker, interval=intraday_interval)

    if raw_intraday.empty:
        st.warning("Data intraday belum tersedia untuk emiten ini (kemungkinan saham tidak likuid atau di luar jam sesi perdagangan).")
    else:
        vol_df = analyze_massive_volume(
            raw_intraday,
            rvol_threshold=rvol_threshold,
            min_value_miliar=min_val_miliar,
        )

        latest_date = vol_df["Date_Only"].iloc[-1]
        today_df = vol_df[vol_df["Date_Only"] == latest_date].copy()
        last_bar = today_df.iloc[-1]
        last_time_str = today_df.index[-1].strftime("%d %b %Y, %H:%M WIB")

        recent_spikes = today_df[today_df["Is_Spike"]]
        if last_bar["Is_Spike"]:
            if "BUY" in last_bar["Sinyal_Volume"]:
                st.success(
                    f"🚨 **TERDETEKSI AKUMULASI MASIF (HAKA) SEKARANG!** ({last_time_str}) — "
                    f"Volume melonjak **{last_bar['RVOL']:.2f}x** dari rata-rata "
                    f"(Est. Nilai: **Rp {last_bar['Est_Value_Rp']/1e9:,.2f} Miliar**, "
                    f"Dominasi Beli: **{last_bar['Buy_Ratio']*100:.1f}%**)."
                )
            elif "SELL" in last_bar["Sinyal_Volume"]:
                st.error(
                    f"🚨 **TERDETEKSI DISTRIBUSI / JUAL MASIF (HAKI) SEKARANG!** ({last_time_str}) — "
                    f"Volume melonjak **{last_bar['RVOL']:.2f}x** dari rata-rata "
                    f"(Est. Nilai: **Rp {last_bar['Est_Value_Rp']/1e9:,.2f} Miliar**, "
                    f"Dominasi Jual: **{(1 - last_bar['Buy_Ratio'])*100:.1f}%**)."
                )
            else:
                st.warning(
                    f"⚠️ **LONJAKAN VOLUME CROSSING TERDETEKSI!** ({last_time_str}) — "
                    f"Volume **{last_bar['RVOL']:.2f}x** rata-rata (Rp {last_bar['Est_Value_Rp']/1e9:,.2f} Miliar) dengan pergerakan harga ketat."
                )
        elif not recent_spikes.empty:
            last_spike = recent_spikes.iloc[-1]
            spike_time = recent_spikes.index[-1].strftime("%H:%M WIB")
            st.info(
                f"📡 **Sinyal Terakhir Hari Ini ({spike_time}):** {last_spike['Sinyal_Volume']} "
                f"sebesar **{last_spike['RVOL']:.2f}x** rata-rata (Est. Rp {last_spike['Est_Value_Rp']/1e9:,.2f} Miliar). "
                f"Bar terakhir ({last_time_str}) terpantau normal ({last_bar['RVOL']:.2f}x)."
            )
        else:
            st.info(
                f"✅ **Aktivitas Volume Terkendali** hingga {last_time_str}. "
                f"Belum terdeteksi lonjakan volume di atas ambang **{rvol_threshold:.1f}x** pada sesi hari ini."
            )

        tot_buy_today = float(today_df["Est_Buy_Vol"].sum())
        tot_sell_today = float(today_df["Est_Sell_Vol"].sum())
        tot_vol_today = tot_buy_today + tot_sell_today
        buy_dom_pct = (tot_buy_today / tot_vol_today * 100) if tot_vol_today > 0 else 50.0
        net_delta_today = tot_buy_today - tot_sell_today

        v_c1, v_c2, v_c3, v_c4 = st.columns(4)
        v_c1.metric(
            f"Volume Bar Terakhir ({today_df.index[-1].strftime('%H:%M')})",
            f"{int(last_bar['Volume']):,} lbr",
            delta=f"RVOL {last_bar['RVOL']:.2f}x",
        )
        v_c2.metric(
            "Est. Nilai Bar Terakhir",
            f"Rp {last_bar['Est_Value_Rp']/1e9:,.2f} M",
            delta=f"Z-Score: {last_bar['Vol_ZScore']:+.2f}",
        )
        v_c3.metric(
            "Dominasi Beli vs Jual Hari Ini",
            f"{buy_dom_pct:.1f}% Buy",
            delta=f"{100 - buy_dom_pct:.1f}% Sell",
            delta_color="off",
        )
        v_c4.metric(
            "Net Volume Delta Hari Ini",
            f"{int(net_delta_today):+,} lbr",
            delta="Net Akumulasi" if net_delta_today > 0 else "Net Distribusi",
        )

        st.plotly_chart(create_volume_radar_chart(vol_df), use_container_width=True)

        st.markdown("#### 📋 Riwayat Lonjakan Volume Masif Terdeteksi")
        all_spikes = vol_df[vol_df["Is_Spike"]].sort_index(ascending=False)
        if not all_spikes.empty:
            log_df = pd.DataFrame({
                "Waktu (WIB)": all_spikes.index.strftime("%Y-%m-%d %H:%M"),
                "Harga (Rp)": all_spikes["Close"],
                "Volume (Lembar)": all_spikes["Volume"],
                "RVOL": all_spikes["RVOL"],
                "Est. Nilai (Rp Miliar)": all_spikes["Est_Value_Rp"] / 1e9,
                "Dominasi Beli (%)": all_spikes["Buy_Ratio"] * 100,
                "Sinyal Deteksi": all_spikes["Sinyal_Volume"],
            })
            st.dataframe(
                log_df.style.format({
                    "Harga (Rp)": "Rp {:,.0f}",
                    "Volume (Lembar)": "{:,.0f}",
                    "RVOL": "{:.2f}x",
                    "Est. Nilai (Rp Miliar)": "Rp {:,.2f} M",
                    "Dominasi Beli (%)": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Tidak ada bar yang melampaui ambang RVOL & minimal nilai transaksi pada rentang intraday ini.")

with tab2:
    st.markdown("### Proyeksi Harga Saham (Historis + Sentimen Berita)")
    st.plotly_chart(
        create_forecast_chart(hist_df, forecast_df, valuation["composite"]),
        use_container_width=True,
    )

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Akurasi Uji Historis (MAPE)", f"{mape:.2f}%")
    c_m2.metric("Rata-rata Error (MAE)", f"Rp {mae:,.0f}")
    c_m3.metric("Sentimen Domestik vs Global", f"ID: {score_id:+.2f} | Global: {score_global:+.2f}")

    with st.expander("Lihat Tabel Proyeksi Harian"):
        disp_forecast = forecast_df.copy()
        disp_forecast.index = disp_forecast.index.strftime("%Y-%m-%d")
        st.dataframe(disp_forecast.style.format("Rp {:,.0f}"), use_container_width=True)

with tab3:
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

with tab4:
    col_id, col_gl = st.columns(2)
    with col_id:
        st.markdown(f"### 🇮🇩 Berita Dalam Negeri (Skor: `{score_id:+.2f}`)")
        if not df_news_id.empty:
            disp_id = df_news_id.copy()
            disp_id["Judul"] = (
                disp_id["Link"] + "#" + disp_id["Judul"].str.replace("#", "", regex=False)
            )
            st.dataframe(
                disp_id[["Judul", "Tanggal", "Sumber", "Skor Sentimen", "Analisis"]],
                column_config={
                    "Judul": st.column_config.LinkColumn("Judul", display_text=r"#([^#]*)$"),
                    "Tanggal": st.column_config.TextColumn("Tanggal Publish"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Tidak ditemukan berita domestik terbaru untuk kata kunci ini.")

    with col_gl:
        st.markdown(f"### 🌍 Berita Luar Negeri / Global (Skor: `{score_global:+.2f}`)")
        if not df_news_global.empty:
            disp_gl = df_news_global.copy()
            disp_gl["Judul"] = (
                disp_gl["Link"] + "#" + disp_gl["Judul"].str.replace("#", "", regex=False)
            )
            st.dataframe(
                disp_gl[["Judul", "Tanggal", "Sumber", "Skor Sentimen", "Analisis"]],
                column_config={
                    "Judul": st.column_config.LinkColumn("Judul", display_text=r"#([^#]*)$"),
                    "Tanggal": st.column_config.TextColumn("Tanggal Publish"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Tidak ditemukan berita internasional terbaru untuk emiten ini.")

st.divider()
st.caption("⚠️ **Disclaimer:** CekWajar adalah alat bantu riset kuantitatif. Hasil valuasi, deteksi volume, dan prediksi Machine Learning bukan merupakan ajakan membeli atau menjual saham (DYOR).")
