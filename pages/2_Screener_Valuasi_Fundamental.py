import pandas as pd
import plotly.express as px
import streamlit as st

from modules import IDX_WATCHLISTS, scan_fundamental_screener

st.title("💎 Screener Saham Undervalued & F-Score Prima")
st.caption("Pemindaian otomatis saham IDX berharga diskon (Margin of Safety tinggi) dengan laporan keuangan sehat (Piotroski F-Score 0–9)")

# ==========================================
# SIDEBAR FILTER SCREENER FUNDAMENTAL
# ==========================================
with st.sidebar:
    st.header("🎛️ Filter Screener Fundamental")
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
    quick_filter = st.radio(
        "Filter Kategori Saham",
        [
            "Semua Emiten Terpindai (Urut Skor Gabungan)",
            "💎 Zona Emas: Undervalued (MoS ≥ 15%) & F-Score ≥ 7",
            "🟢 Hanya Saham Undervalued (MoS ≥ 15%)",
            "🛡️ Hanya Fundamental Prima (F-Score ≥ 7)",
        ],
        index=0,
    )
    min_mos = st.slider("Minimal Margin of Safety / MoS (%)", min_value=-50, max_value=50, value=-20, step=5)
    min_fscore = st.slider("Minimal Piotroski F-Score (0–9)", min_value=0, max_value=9, value=4, step=1)

    st.divider()
    st.subheader("📐 Asumsi Valuasi Global")
    fair_per = st.slider("Target PER Wajar (x)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
    fair_pbv = st.slider("Target PBV Wajar (x)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    discount_rate_pct = st.slider("Tingkat Diskonto / Required Return (%)", min_value=8.0, max_value=16.0, value=11.0, step=0.5)

target_tickers = list(IDX_WATCHLISTS[group_name])
if extra_tickers.strip():
    custom_list = [x.strip().upper() for x in extra_tickers.split(",") if x.strip()]
    target_tickers.extend(custom_list)
target_tickers = list(dict.fromkeys(target_tickers))

# ==========================================
# EKSEKUSI PEMINDAIAN FUNDAMENTAL
# ==========================================
top_c1, top_c2 = st.columns([4, 1])
with top_c1:
    st.markdown(
        f"**Cakupan:** `{len(target_tickers)} Emiten IDX` | "
        f"**Asumsi:** `PER {fair_per:.1f}x` · `PBV {fair_pbv:.1f}x` · `Discount Rate {discount_rate_pct:.1f}%`"
    )
with top_c2:
    if st.button("🔄 Scan Ulang Fundamental", use_container_width=True):
        scan_fundamental_screener.clear()
        st.rerun()

with st.spinner(f"Menganalisis laporan keuangan & menghitung harga wajar {len(target_tickers)} emiten IDX..."):
    df_fund, details_map = scan_fundamental_screener(
        tickers_tuple=tuple(target_tickers),
        fair_per=fair_per,
        fair_pbv=fair_pbv,
        discount_rate=discount_rate_pct / 100.0,
    )

if df_fund.empty:
    st.warning("Data fundamental belum berhasil ditarik saat ini. Silakan coba beberapa saat lagi.")
    st.stop()

# ==========================================
# KPI RINGKASAN HASIL SCREENER
# ==========================================
valid_df = df_fund.dropna(subset=["Margin of Safety (%)", "F-Score (/9)"]).copy()
gems_df = valid_df[(valid_df["Margin of Safety (%)"] >= 15) & (valid_df["F-Score (/9)"] >= 7)]
underval_df = valid_df[valid_df["Margin of Safety (%)"] >= 15]
healthy_df = valid_df[valid_df["F-Score (/9)"] >= 7]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Emiten Berhasil Dipindai", f"{len(df_fund)} Saham")
k2.metric(
    "💎 Zona Emas (Murah & Sehat)",
    f"{len(gems_df)} Saham",
    delta=", ".join(gems_df["Kode"].head(4).tolist()) if not gems_df.empty else "Belum ada",
)
k3.metric(
    "🟢 Undervalued (MoS ≥ 15%)",
    f"{len(underval_df)} Saham",
    delta=f"Tertinggi: {underval_df.iloc[0]['Kode']} ({underval_df.iloc[0]['Margin of Safety (%)']:+.1f}%)" if not underval_df.empty else "Belum ada",
)
k4.metric(
    "🛡️ F-Score Prima (≥ 7/9)",
    f"{len(healthy_df)} Saham",
    delta=", ".join(healthy_df["Kode"].head(4).tolist()) if not healthy_df.empty else "Belum ada",
)

# ==========================================
# GRAFIK KUADRAN: PIOTROSKI F-SCORE VS MARGIN OF SAFETY
# ==========================================
if not valid_df.empty:
    plot_q = valid_df.copy()
    plot_q["MoS_Tampilan"] = plot_q["Margin of Safety (%)"].clip(-60, 80)

    fig_quad = px.scatter(
        plot_q,
        x="F-Score (/9)",
        y="MoS_Tampilan",
        text="Kode",
        color="Kuadran Analisa",
        color_discrete_map={
            "💎 Zona Emas (Undervalued & Sehat)": "#00FF7F",
            "⚠️ Waspada Value Trap (Murah, F-Score < 7)": "#FFD700",
            "🔵 Saham Prima Harga Wajar/Premium": "#00BFFF",
            "⚪ Netral / Kurang Menarik": "#8899A6",
        },
        hover_data={
            "Kode": True,
            "Emiten": True,
            "Harga (Rp)": ":,.0f",
            "Harga Wajar (Rp)": ":,.0f",
            "Margin of Safety (%)": ":+.1f",
            "F-Score (/9)": True,
            "PER (x)": ":.2f",
            "PBV (x)": ":.2f",
            "MoS_Tampilan": False,
        },
        title="Peta Kuadran Value Investing: Kesehatan Laporan (F-Score) vs. Diskon Harga (MoS %)",
    )
    fig_quad.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
    fig_quad.add_hline(y=15, line_dash="dash", line_color="#00FF7F", annotation_text="Batas Undervalued (MoS 15%)", annotation_position="top left")
    fig_quad.add_vline(x=6.5, line_dash="dash", line_color="cyan", annotation_text="Batas Fundamental Sehat (F-Score ≥ 7)", annotation_position="top right")
    fig_quad.update_layout(
        height=440,
        template="plotly_dark",
        xaxis_title="Piotroski F-Score (0 = Lemah, 9 = Sangat Sehat)",
        yaxis_title="Margin of Safety / MoS (%)",
        xaxis=dict(range=[-0.5, 9.8], dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_quad, use_container_width=True)

# ==========================================
# TABEL HASIL PEMINDAIAN
# ==========================================
filtered = df_fund.copy()
if quick_filter == "💎 Zona Emas: Undervalued (MoS ≥ 15%) & F-Score ≥ 7":
    filtered = filtered[(filtered["Margin of Safety (%)"].fillna(-999) >= 15) & (filtered["F-Score (/9)"].fillna(0) >= 7)]
elif quick_filter == "🟢 Hanya Saham Undervalued (MoS ≥ 15%)":
    filtered = filtered[filtered["Margin of Safety (%)"].fillna(-999) >= 15]
elif quick_filter == "🛡️ Hanya Fundamental Prima (F-Score ≥ 7)":
    filtered = filtered[filtered["F-Score (/9)"].fillna(0) >= 7]

filtered = filtered[
    (filtered["Margin of Safety (%)"].fillna(-999) >= min_mos)
    & (filtered["F-Score (/9)"].fillna(0) >= min_fscore)
].copy()

st.markdown(f"### 📋 Daftar Hasil Screener Valuasi & Kesehatan ({len(filtered)} Emiten)")

if filtered.empty:
    st.info("Tidak ada emiten yang memenuhi kriteria filter tersebut. Coba turunkan batas minimal MoS atau F-Score di sidebar.")
else:
    def style_mos(val):
        if pd.isna(val):
            return "inherit"
        color = "#FF4B4B" if val < 0 else ("#00CC96" if val >= 15 else "#FFD700")
        return f"color: {color}; font-weight: bold;"

    st.dataframe(
        filtered.style.format(
            {
                "Harga (Rp)": "Rp {:,.0f}",
                "Harga Wajar (Rp)": "Rp {:,.0f}",
                "Margin of Safety (%)": "{:+.1f}%",
                "F-Score (/9)": "{:.0f} / 9",
                "Skor Gabungan": "{:.1f}",
                "PER (x)": "{:.2f}x",
                "PBV (x)": "{:.2f}x",
                "ROE (%)": "{:.1f}%",
            },
            na_rep="-",
        ).map(style_mos, subset=["Margin of Safety (%)"]),
        use_container_width=True,
        hide_index=True,
    )

# ==========================================
# BEDAH CEPAT RINCIAN VALUASI & F-SCORE EMITEN
# ==========================================
st.divider()
st.subheader("🔍 Bedah Cepat Valuasi & 9 Kriteria F-Score Emiten Terpilih")
inspect_list = filtered["Kode"].tolist() if not filtered.empty else df_fund["Kode"].tolist()
selected_code = st.selectbox(
    "Pilih kode saham dari daftar di atas untuk melihat rincian bobot harga wajar & laporan keuangannya:",
    inspect_list,
    index=0,
)

if selected_code and selected_code in details_map:
    detail = details_map[selected_code]
    val_info = detail["valuation"]
    pio_info = detail["piotroski"]

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"#### 💰 Rincian Valuasi ({detail['company_name']})")
        st.caption(f"Profil Sektor: **{val_info['profile']}**")
        if val_info.get("details"):
            st.dataframe(
                pd.DataFrame(val_info["details"]).style.format({
                    "Harga Wajar (Rp)": "Rp {:,.0f}",
                    "Bobot (%)": "{:.1f}%",
                    "Kontribusi (Rp)": "Rp {:,.0f}",
                    "Potensi Upside/Downside (%)": "{:+.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Data EPS/BVPS tidak mencukupi untuk perhitungan multi-model.")

    with c_right:
        st.markdown(f"#### 🛡️ Rincian Piotroski F-Score ({pio_info.get('score_9', 0)}/9)")
        st.caption(f"Status: **{pio_info.get('verdict', '-')}**")
        if pio_info.get("criteria"):
            st.dataframe(
                pd.DataFrame(pio_info["criteria"]),
                use_container_width=True,
                hide_index=True,
            )
