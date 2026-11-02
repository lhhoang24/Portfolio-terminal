from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


_C = {
    "bg": "#000000",
    "card": "#0B1117",
    "grid": "rgba(154, 170, 188, 0.10)",
    "text": "#F4F7FB",
    "muted": "#9AAABC",
    "accent": "#1685FF",
    "positive": "#00D68F",
    "negative": "#FF5263",
    "border": "rgba(154, 170, 188, 0.22)",
}


_PALETTE = [
    "#1685FF", "#7F93A8", "#FFB020", "#00D68F", "#A855F7",
    "#37B9FF", "#FF5263", "#F97316", "#EC4899", "#84CC16",
]

_FONT = dict(
    family="Inter, system-ui, -apple-system, Segoe UI, sans-serif",
    color=_C["muted"],
    size=12,
)


_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=_FONT,
    title=dict(font=dict(size=14, color=_C["text"]), x=0, xanchor="left"),
    colorway=_PALETTE,
    xaxis=dict(
        gridcolor=_C["grid"],
        zerolinecolor=_C["grid"],
        linecolor=_C["border"],
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor=_C["grid"],
        zerolinecolor=_C["grid"],
        linecolor=_C["border"],
        tickfont=dict(size=11),
    ),
    legend=dict(font=dict(size=11, color=_C["muted"])),
    hoverlabel=dict(
        bgcolor="#0B1117",
        bordercolor="#26384a",
        font=dict(size=12, color=_C["text"]),
    ),
)
pio.templates["terminal"] = _tpl
pio.templates.default = "terminal"

_BASE = dict(
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(orientation="h", y=-0.15, xanchor="left", x=0),
    hovermode="x unified",
    height=400,
)


_SHARPE_SCALE = [[0, "#12263d"], [0.5, "#1685FF"], [1, "#9ADBFF"]]


def line_fig(
    df: pd.DataFrame | pd.Series,
    title: str,
    pct: bool = False,
    yaxis_title: str | None = None,
    hline: float | None = None,
) -> go.Figure:
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name or "value")

    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=str(col),
                mode="lines",
                line=dict(width=2, color=_PALETTE[i % len(_PALETTE)]),
            )
        )

    fig.update_layout(**{
        **_BASE,
        "title": title,
        "yaxis_title": yaxis_title,
    })

    if pct:
        fig.update_yaxes(tickformat=".0%")

    if hline is not None:
        fig.add_hline(
            y=hline,
            line_dash="dot",
            line_color=_C["muted"],
            opacity=0.45,
        )
    return fig


def pie_fig(weights: pd.Series, title: str, center_text: str | None = None) -> go.Figure:
    w = weights[weights > 1e-6]
    fig = go.Figure(
        go.Pie(
            labels=w.index,
            values=w.values,
            hole=0.68,
            marker=dict(
                colors=_PALETTE[:len(w)],
                line=dict(color=_C["bg"], width=2),
            ),
            textinfo="none",
            hovertemplate="%{label}: %{percent}<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left",
                    font=dict(size=12, color=_C["text"])),
        margin=dict(l=16, r=16, t=56, b=24),
        height=380,
    )
    if center_text:
        fig.add_annotation(text=center_text, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper",
                           font=dict(size=13, color=_C["muted"]), align="center")
    return fig


def perf_fig(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        y = df[col] - 1.0
        last = y.dropna().iloc[-1] if y.notna().any() else float("nan")
        name = f"{col}  {last:+.1%}" if np.isfinite(last) else str(col)
        kw = dict(fill="tozeroy", fillcolor="rgba(22,133,255,0.08)") if i == 0 else {}
        fig.add_trace(go.Scatter(x=df.index, y=y, name=name, mode="lines",
                                 line=dict(width=2.2 if i == 0 else 1.6, color=_PALETTE[i % len(_PALETTE)]),
                                 hovertemplate="%{y:+.1%}", **kw))
    fig.update_layout(**{**_BASE, "title": title, "height": 380})
    fig.update_yaxes(tickformat=".0%", zeroline=True)
    return fig


def corr_heatmap(
    corr: pd.DataFrame,
    title: str = "Ma trận tương quan",
) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, "#FF5263"], [0.5, "#0B1117"], [1, "#1685FF"]],
        zmin=-1,
        zmax=1,
        title=title,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=16, r=16, t=48, b=16), height=420)
    fig.update_traces(textfont=dict(size=11))
    return fig


def hist_fig(
    r: pd.Series,
    title: str,
    var_line: float | None = None,
) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=r,
            nbinsx=50,
            marker=dict(
                color=_C["accent"],
                line=dict(color=_C["bg"], width=1),
            ),
        )
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis_title="Lợi suất ngày",
        yaxis_title="Số phiên",
        margin=dict(l=16, r=16, t=48, b=16),
        height=380,
    )
    fig.update_xaxes(tickformat=".1%")

    if var_line is not None and np.isfinite(var_line):
        fig.add_vline(
            x=var_line,
            line_dash="dash",
            line_color=_C["negative"],
            annotation_text="VaR 95%",
            annotation_font=dict(color=_C["negative"], size=11),
        )
    return fig


def candlestick_fig(
    ohlc: pd.DataFrame,
    ticker: str,
    overlays: dict[str, pd.Series],
    kind: str = "Candlestick",
) -> go.Figure:
    fig = go.Figure()

    if kind == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"],
                name=ticker,
                increasing=dict(
                    line=dict(color=_C["positive"]),
                    fillcolor=_C["positive"],
                ),
                decreasing=dict(
                    line=dict(color=_C["negative"]),
                    fillcolor=_C["negative"],
                ),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=ohlc.index,
                y=ohlc["Close"],
                name=ticker,
                mode="lines",
                line=dict(color=_PALETTE[0], width=2),
            )
        )

    for i, (name, series) in enumerate(overlays.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=name,
                mode="lines",
                line=dict(
                    width=1.5,
                    dash="dot",
                    color=_PALETTE[(i + 1) % len(_PALETTE)],
                ),
            )
        )


    layout = {
        **_BASE,
        "title": f"{ticker} – giá",
        "height": 520,
        "xaxis_rangeslider_visible": False,
    }
    fig.update_layout(**layout)
    return fig


def frontier_fig(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    points: dict[str, tuple[float, float]],
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=frontier["Biến động"],
            y=frontier["Lợi suất"],
            mode="markers",
            marker=dict(
                size=3,
                color=frontier["Sharpe"],
                colorscale=_SHARPE_SCALE,
                showscale=True,
                opacity=0.6,

                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(size=11)),
                    tickfont=dict(size=10),
                ),
            ),
            name="Danh mục ngẫu nhiên",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=assets["Biến động"],
            y=assets["Lợi suất"],
            mode="markers+text",
            text=assets.index,
            textposition="top center",
            marker=dict(
                size=9,
                color=_C["text"],
                symbol="diamond",
                line=dict(width=1, color=_C["bg"]),
            ),
            textfont=dict(size=11, color=_C["text"]),
            name="Tài sản đơn lẻ",
        )
    )

    _sym = {
        "Max Sharpe": "star",
        "Min Variance": "square",
        "Danh mục hiện tại": "circle",
    }
    _clr = {
        "Max Sharpe": "#FFB020",
        "Min Variance": "#00D68F",
        "Danh mục hiện tại": "#A855F7",
    }

    for name, (vol, ret) in points.items():
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers",
                name=name,
                marker=dict(
                    size=14,
                    symbol=_sym.get(name, "star"),
                    color=_clr.get(name, "#FFB020"),
                    line=dict(width=2, color=_C["bg"]),
                ),
            )
        )

    fig.update_layout(
        title="Đường biên hiệu quả (Monte Carlo)",
        xaxis_title="Biến động năm",
        yaxis_title="Lợi suất kỳ vọng năm",
        height=480,
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def beta_heatmap(beta_df: pd.DataFrame, title: str) -> go.Figure:
    df = beta_df.dropna(how="all")
    fig = px.imshow(
        df.T,
        aspect="auto",
        color_continuous_scale=_SHARPE_SCALE,
        title=title,
    )
    fig.update_layout(margin=dict(l=16, r=16, t=48, b=16), height=380)
    return fig


def bar_fig(s: pd.Series, title: str, pct: bool = True) -> go.Figure:
    colors = [
        _C["positive"] if value >= 0 else _C["negative"]
        for value in s.values
    ]
    fig = go.Figure(
        go.Bar(
            x=s.index,
            y=s.values,
            marker=dict(
                color=colors,
                line=dict(color=_C["bg"], width=1),
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=16, r=16, t=48, b=16),
        height=380,
    )
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return fig


def risk_return_scatter(df: pd.DataFrame, title: str) -> go.Figure:
    d = df.reset_index().rename(
        columns={df.index.name or "index": "Mã"}
    )
    fig = px.scatter(
        d,
        x="Biến động năm",
        y="Lợi suất năm hóa",
        color="Sharpe",
        text="Mã",
        color_continuous_scale=_SHARPE_SCALE,
        title=title,
        hover_data=["Beta", "Max Drawdown"],
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(
            size=10,
            line=dict(width=1, color=_C["bg"]),
        ),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=520, margin=dict(l=16, r=16, t=48, b=16))
    return fig


def regime_fig(score: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=score.index,
            y=score.values,
            mode="lines",
            line_shape="hv",
            name="Regime score",
            line=dict(color=_PALETTE[0], width=2),
        )
    )
    fig.add_hrect(
        y0=-0.2,
        y1=0.5,
        fillcolor=_C["negative"],
        opacity=0.07,
        line_width=0,
        annotation_text="Bear",
        annotation_position="left",
        annotation_font=dict(size=10, color=_C["negative"]),
    )
    fig.add_hrect(
        y0=0.5,
        y1=1.5,
        fillcolor=_C["muted"],
        opacity=0.04,
        line_width=0,
        annotation_text="Neutral",
        annotation_position="left",
        annotation_font=dict(size=10, color=_C["muted"]),
    )
    fig.add_hrect(
        y0=1.5,
        y1=3.2,
        fillcolor=_C["positive"],
        opacity=0.07,
        line_width=0,
        annotation_text="Bull",
        annotation_position="left",
        annotation_font=dict(size=10, color=_C["positive"]),
    )
    fig.update_layout(
        title="Điểm regime (0=Bear, 1=Neutral, 2–3=Bull)",
        yaxis=dict(dtick=1, range=[-0.2, 3.2]),
        margin=dict(l=16, r=16, t=48, b=16),
        hovermode="x unified",
        height=350,
    )
    return fig


def regime_price_map(price: pd.Series, regime: pd.Series, title: str) -> go.Figure:
    colors = {"Bull": _C["positive"], "Neutral": "#FFB020", "Bear": _C["negative"]}
    df = pd.concat([price.rename("price"), regime.rename("regime")], axis=1).dropna()
    fig = go.Figure()
    for name in ("Bull", "Neutral", "Bear"):
        d = df[df["regime"] == name]
        fig.add_trace(go.Scatter(x=d.index, y=d["price"], mode="markers", name=name,
                                 marker=dict(size=4, color=colors[name])))
    fig.update_layout(**{**_BASE, "title": title, "yaxis_title": "Giá", "hovermode": "closest"})
    return fig


def factor_beta_fig(betas: pd.Series, title: str) -> go.Figure:
    fig = bar_fig(betas, title, pct=False)
    fig.update_traces(text=[f"{v:.2f}" for v in betas.values], textposition="outside",
                      hovertemplate="%{x}: %{y:.2f}<extra></extra>")
    fig.add_hline(y=0, line_color=_C["muted"], opacity=0.35, line_width=1)
    return fig
