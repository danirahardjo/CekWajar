import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_rsi_chart(
    rsi_df: pd.DataFrame,
    overbought: int = 70,
    oversold: int = 30,
) -> go.Figure:
    """Membuat grafik 2-panel Candlestick + EMA-20 serta Osilator RSI & Signal Line (MA-9)."""
    plot_df = rsi_df.iloc[-120:].copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Pergerakan Harga, EMA-20 & Titik Sinyal Entry RSI",
            "Osilator Relative Strength Index (RSI) & Garis Sinyal (MA-9)",
        ),
    )

    # Panel 1: Candlestick + EMA-20
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Harga",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["EMA_20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="orange", width=1.6),
        ),
        row=1,
        col=1,
    )

    buy_pts = plot_df[plot_df["RSI_Signal"].str.startswith("LAYAK BELI")]
    ob_pts = plot_df[plot_df["RSI_Signal"].str.startswith("HINDARI")]

    if not buy_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_pts.index,
                y=buy_pts["Low"] * 0.985,
                mode="markers",
                name="Sinyal Layak Beli RSI",
                marker=dict(symbol="triangle-up", size=12, color="#00FF7F", line=dict(width=1, color="black")),
            ),
            row=1,
            col=1,
        )

    if not ob_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=ob_pts.index,
                y=ob_pts["High"] * 1.015,
                mode="markers",
                name="Masuk Overbought (≥ 70)",
                marker=dict(symbol="triangle-down", size=12, color="#FF4B4B", line=dict(width=1, color="black")),
            ),
            row=1,
            col=1,
        )

    # Panel 2: Garis RSI & RSI MA-9
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["RSI"],
            mode="lines",
            name="RSI (14)",
            line=dict(color="#00E5FF", width=2.2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["RSI_MA"],
            mode="lines",
            name="RSI Signal (MA-9)",
            line=dict(color="#FFD700", width=1.5, dash="dot"),
        ),
        row=2,
        col=1,
    )

    # Zona Overbought (≥ 70), Zona Sehat (45–65), dan Oversold (≤ 30)
    fig.add_hrect(y0=overbought, y1=100, fillcolor="rgba(255, 75, 75, 0.14)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=45, y1=65, fillcolor="rgba(0, 229, 255, 0.07)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=oversold, fillcolor="rgba(0, 255, 127, 0.14)", line_width=0, row=2, col=1)

    fig.add_hline(y=overbought, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Overbought ({overbought})", row=2, col=1)
    fig.add_hline(y=55, line_dash="dot", line_color="rgba(200,200,200,0.35)", row=2, col=1)
    fig.add_hline(y=45, line_dash="dot", line_color="rgba(200,200,200,0.35)", row=2, col=1)
    fig.add_hline(y=oversold, line_dash="dash", line_color="#00FF7F", annotation_text=f"Oversold ({oversold})", row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
    fig.update_yaxes(title_text="Nilai RSI (0–100)", range=[0, 100], row=2, col=1)
    return fig

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
