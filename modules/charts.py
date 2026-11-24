import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_volume_radar_chart(vol_df: pd.DataFrame) -> go.Figure:
    """Membuat grafik 2-panel untuk pergerakan harga intraday, VWAP, dan titik volume masif."""
    plot_df = vol_df.iloc[-120:].copy()
    fig_vol = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            "Pergerakan Harga Intraday, VWAP & Titik Volume Masif",
            "Volume Transaksi & Rata-Rata (MA20)",
        ),
    )

    fig_vol.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Harga Intraday",
        ),
        row=1,
        col=1,
    )
    fig_vol.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["VWAP"],
            mode="lines",
            name="VWAP Harian",
            line=dict(color="#FFD700", width=1.6),
        ),
        row=1,
        col=1,
    )

    buy_spikes = plot_df[plot_df["Sinyal_Volume"].str.contains("BUY")]
    sell_spikes = plot_df[plot_df["Sinyal_Volume"].str.contains("SELL")]

    if not buy_spikes.empty:
        fig_vol.add_trace(
            go.Scatter(
                x=buy_spikes.index,
                y=buy_spikes["Low"] * 0.997,
                mode="markers",
                name="Masif Buy (HAKA)",
                marker=dict(symbol="triangle-up", size=13, color="#00FF7F", line=dict(width=1, color="black")),
            ),
            row=1,
            col=1,
        )

    if not sell_spikes.empty:
        fig_vol.add_trace(
            go.Scatter(
                x=sell_spikes.index,
                y=sell_spikes["High"] * 1.003,
                mode="markers",
                name="Masif Sell (HAKI)",
                marker=dict(symbol="triangle-down", size=13, color="#FF4B4B", line=dict(width=1, color="black")),
            ),
            row=1,
            col=1,
        )

    bar_colors = []
    for _, r in plot_df.iterrows():
        if "BUY" in r["Sinyal_Volume"]:
            bar_colors.append("#00FF7F")
        elif "SELL" in r["Sinyal_Volume"]:
            bar_colors.append("#FF4B4B")
        elif "CROSSING" in r["Sinyal_Volume"]:
            bar_colors.append("#FFD700")
        else:
            bar_colors.append("rgba(150, 160, 180, 0.45)")

    fig_vol.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["Volume"],
            name="Volume",
            marker_color=bar_colors,
        ),
        row=2,
        col=1,
    )
    fig_vol.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Vol_MA20"],
            mode="lines",
            name="Rata-rata Vol (MA20)",
            line=dict(color="cyan", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig_vol.update_layout(
        height=580,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig_vol.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
    fig_vol.update_yaxes(title_text="Volume (Lembar)", row=2, col=1)
    return fig_vol


def create_forecast_chart(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    fair_value: float,
) -> go.Figure:
    """Membuat grafik candlestick historis, kurva proyeksi AI, dan garis batas harga wajar."""
    recent_hist = hist_df.iloc[-120:].copy()
    recent_hist["MA20"] = recent_hist["Close"].rolling(20).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=recent_hist.index,
            open=recent_hist["Open"],
            high=recent_hist["High"],
            low=recent_hist["Low"],
            close=recent_hist["Close"],
            name="Harga Historis",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=recent_hist.index,
            y=recent_hist["MA20"],
            mode="lines",
            name="MA 20 Hari",
            line=dict(color="orange", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Prediksi_Harga"],
            mode="lines+markers",
            name="Prediksi CekWajar",
            line=dict(color="#00CC96", width=2.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df["Batas_Atas"]) + list(forecast_df["Batas_Bawah"][::-1]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Rentang Volatilitas Prediksi",
        )
    )

    if fair_value > 0:
        fig.add_hline(
            y=fair_value,
            line_dash="dot",
            line_color="cyan",
            annotation_text=f"Harga Wajar: Rp {fair_value:,.0f}",
            annotation_position="top left",
        )

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        yaxis_title="Harga Saham (IDR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
