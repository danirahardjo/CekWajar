import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import (
    IDX_WATCHLISTS,
    create_volume_radar_chart,
    scan_market_volume_screener,
)

st.set_page_config(
    page_title="Radar Volume Pasar IDX | CekWajar Screener",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📡 Radar Volume Pasar IDX (Auto-Screener)")
st.caption("Pemindaian otomatis lonjakan volume jual/beli masif (HAKA / HAKI) lintas emiten tanpa perlu memilih kode saham terlebih dahulu")

# ==========================================
# SIDEBAR FILTER SCREENER
# ==========================================
with st.sidebar:
    st.header("🎛️ Filter Pemindaian Pasar")
    group_name = st.selectbox(
        "Kelompok Saham Dipindai",
        list(IDX_WATCHLISTS.keys()),
        index=0,
    )
    extra_tickers = st.text_input(
        "Tambah Kode Emiten Lain (Opsional, pisahkan koma)",
        placeholder="Contoh: BUMI, PTRO, BRMS",
    )

    st.divider()
    intraday_interval = st.selectbox("Interval Candle Intraday", ["5m", "15m", "1m"], index=0)
    rvol_threshold = st.slider("Ambang Lonjakan Volume (RVOL)", min_value=1.5, max_value=5.0, value=2.2, step=0.1)
    min_val_miliar = st.number_input("Minimal Nilai Transaksi/Bar (Rp Miliar)", min_value=0.0, value=0.5, step=0.5)

    st.divider()
    signal_filter = st.radio(
        "Tampilkan Sinyal",
        [
            "Semua Emiten Dipindai (Urut RVOL)",
            "Hanya Lonjakan Masif (HAKA / HAKI / Crossing)",
            "Hanya Masif Buy / Akumulasi (HAKA) 🟢",
            "Hanya Masif Sell / Distribusi (HAKI) 🔴",
        ],
        index=0,
    )

# Gabungkan daftar watchlist + emiten tambahan jika diisi
target_tickers = list(IDX_WATCHLISTS[group_name])
if extra_tickers.strip():
    custom_list = [x.strip().upper() for x in extra_tickers.split(",") if x.strip()]
    target_tickers.extend(custom_list)
target_tickers = list(dict.fromkeys(target_tickers))

# ==========================================
# EKSEKUSI PEMINDAIAN OTOMATIS
# ==========================================
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.markdown(f"**Cakupan Pemindaian:** `{len(target_tickers)} Emiten` | **Interval:** `{intraday_interval}` | **Ambang RVOL:** `≥ {rvol_threshold:.1f}x`")
with top_col2:
    if st.button("🔄 Scan Ulang Pasar", use_container_width=True):
        scan_market_volume_screener.clear()
        st.rerun()

with st.spinner(f"Memindai aktivitas volume transaksi {len(target_tickers)} emiten IDX secara paralel..."):
    df_scan, analyzed_map = scan_market_volume_screener(
        tickers_tuple=tuple(target_tickers),
        interval=intraday_interval,
        rvol_threshold=rvol_threshold,
        min_value_miliar=min_val_miliar,
    )

if df_scan.empty:
    st.warning("Data intraday belum tersedia saat ini (kemungkinan di luar jam perdagangan bursa atau gangguan koneksi Yahoo Finance).")
    st.stop()

# ==========================================
# KPI RINGKASAN PASAR
# ==========================================
haka_df = df_scan[df_scan["Sinyal Terdeteksi"].str.contains("BUY")]
haki_df = df_scan[df_scan["Sinyal Terdeteksi"].str.contains("SELL")]
top_rvol_row = df_scan.iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Emiten Aktif Terpantau", f"{len(df_scan)} Saham")
k2.metric(
    "Sinyal Akumulasi (HAKA) 🟢",
    f"{len(haka_df)} Saham",
    delta=", ".join(haka_df["Kode"].head(4).tolist()) if not haka_df.empty else "Belum ada",
)
k3.metric(
    "Sinyal Distribusi (HAKI) 🔴",
    f"{len(haki_df)} Saham",
    delta=", ".join(haki_df["Kode"].head(4).tolist()) if not haki_df.empty else "Belum ada",
    delta_color="inverse",
)
k4.metric(
    "Lonjakan RVOL Tertinggi",
    f"{top_rvol_row['Kode']} ({top_rvol_row['RVOL']:.2f}x)",
    delta=f"Rp {top_rvol_row['Est. Nilai Bar (Rp M)']:,.2f} M",
)

# ==========================================
# GRAFIK TOP 15 RVOL PASAR
# ==========================================
top15 = df_scan.head(15).copy()
bar_colors = []
for sig in top15["Sinyal Terdeteksi"]:
    if "BUY" in sig:
        bar_colors.append("#00FF7F")
    elif "SELL" in sig:
        bar_colors.append("#FF4B4B")
    elif "CROSSING" in sig:
        bar_colors.append("#FFD700")
    else:
        bar_colors.append("rgba(130, 150, 180, 0.65)")

fig_rank = go.Figure(
    go.Bar(
        x=top15["Kode"],
        y=top15["RVOL"],
        marker_color=bar_colors,
        text=[f"{v:.2f}x" for v in top15["RVOL"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "RVOL: %{y:.2f}x<br>"
            "Harga: Rp %{customdata[0]:,.0f} (%{customdata[1]:+.2f}%)<br>"
            "Est. Nilai Bar: Rp %{customdata[2]:,.2f} M<br>"
            "Sinyal: %{customdata[3]}<extra></extra>"
        ),
        customdata=top15[["Harga (Rp)", "Perubahan (%)", "Est. Nilai Bar (Rp M)", "Sinyal Terdeteksi"]].values,
    )
)
fig_rank.add_hline(
    y=rvol_threshold,
    line_dash="dash",
    line_color="cyan",
    annotation_text=f"Ambang Masif ({rvol_threshold:.1f}x)",
    annotation_position="top right",
)
fig_rank.update_layout(
    title="Top 15 Emiten dengan Lonjakan Relative Volume (RVOL) Tertinggi Saat Ini",
    height=360,
    template="plotly_dark",
    yaxis_title="RVOL (x Rata-Rata)",
    xaxis_title="Kode Saham IDX",
    margin=dict(t=50, b=30),
)
st.plotly_chart(fig_rank, use_container_width=True)

# ==========================================
# TABEL HASIL SCREENER
# ==========================================
filtered_df = df_scan.copy()
if signal_filter == "Hanya Lonjakan Masif (HAKA / HAKI / Crossing)":
    filtered_df = filtered_df[filtered_df["Sinyal Terdeteksi"] != "Normal"]
elif signal_filter == "Hanya Masif Buy / Akumulasi (HAKA) 🟢":
    filtered_df = filtered_df[filtered_df["Sinyal Terdeteksi"].str.contains("BUY")]
elif signal_filter == "Hanya Masif Sell / Distribusi (HAKI) 🔴":
    filtered_df = filtered_df[filtered_df["Sinyal Terdeteksi"].str.contains("SELL")]

st.markdown(f"### 📋 Daftar Sinyal Volume Real-Time ({len(filtered_df)} Emiten)")

if filtered_df.empty:
    st.info("Tidak ada emiten yang memenuhi kriteria filter sinyal tersebut saat ini. Coba turunkan ambang RVOL di sidebar.")
else:
    def style_change_and_flow(val):
        color = "#FF4B4B" if val < 0 else ("#00CC96" if val > 0 else "inherit")
        return f"color: {color}; font-weight: bold;"

    st.dataframe(
        filtered_df.style.format({
            "Harga (Rp)": "Rp {:,.0f}",
            "Perubahan (%)": "{:+.2f}%",
            "RVOL": "{:.2f}x",
            "Z-Score": "{:+.2f}",
            "Est. Nilai Bar (Rp M)": "Rp {:,.2f} M",
            "Dominasi Beli (%)": "{:.1f}%",
            "Net Flow Hari Ini (Rp M)": "{:+.2f} M",
        }).map(style_change_and_flow, subset=["Perubahan (%)", "Net Flow Hari Ini (Rp M)"]),
        use_container_width=True,
        hide_index=True,
    )

# ==========================================
# INSPEKSI CEPAT GRAFIK INTRADAY (DRILL-DOWN)
# ==========================================
st.divider()
st.subheader("🔍 Inspeksi Cepat Grafik Intraday Emiten Terdeteksi")
inspect_options = filtered_df["Kode"].tolist() if not filtered_df.empty else df_scan["Kode"].tolist()
selected_inspect = st.selectbox(
    "Pilih saham dari hasil pemindaian di atas untuk melihat detail candlestick, VWAP & titik HAKA/HAKI:",
    inspect_options,
    index=0,
)

if selected_inspect and selected_inspect in analyzed_map:
    st.plotly_chart(
        create_volume_radar_chart(analyzed_map[selected_inspect]),
        use_container_width=True,
    )
