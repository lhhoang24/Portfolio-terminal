from __future__ import annotations

import html
import math

import numpy as np


COLORS = {
    "bg": "#000000",
    "sidebar": "#030609",
    "panel": "#070B0F",
    "card": "#0B1117",
    "border": "#17232D",
    "blue": "#1685FF",
    "cyan": "#37B9FF",
    "green": "#00D68F",
    "red": "#FF5263",
    "purple": "#A855F7",
    "amber": "#FFB020",
    "text": "#F4F7FB",
    "text2": "#9AAABC",
    "muted": "#6D7C8D",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #000000; --sidebar: #030609; --panel: #070B0F; --card: #0B1117; --line: #17232D;
    --blue: #1685FF; --cyan: #37B9FF; --green: #00D68F; --red: #FF5263;
    --purple: #A855F7; --amber: #FFB020;
    --ink: #F4F7FB; --slate: #9AAABC; --faint: #6D7C8D;
    --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stMarkdown, button, input, textarea {
    font-family: var(--sans) !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
[class*="material-symbols"], [data-testid="stExpanderToggleIcon"], summary span:first-child {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp { background: var(--bg); }
header[data-testid="stHeader"], .stAppHeader { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 3.25rem !important; padding-bottom: 3rem; max-width: 1480px; }
h1, h2, h3, h4, h5 { color: var(--ink); letter-spacing: -0.01em; }
:focus-visible { outline: 2px solid var(--blue) !important; outline-offset: 2px; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #1d2c38; border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }

[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 0.72rem; font-weight: 600; color: var(--faint); text-transform: uppercase; letter-spacing: 0.08em;
    margin: 1.3rem 0 0.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3:first-of-type { border-top: 0; padding-top: 0; margin-top: 0.4rem; }
[data-testid="stSidebar"] label p { color: var(--slate); font-size: 0.82rem; }

.brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0 1rem; margin-bottom: 0.2rem; border-bottom: 1px solid var(--line); }
.brand-mark { flex: none; width: 2.3rem; height: 2.3rem; border-radius: 10px; background: rgba(22,133,255,0.14);
              border: 1px solid rgba(22,133,255,0.45); display: grid; place-items: center; }
.brand-mark svg { width: 1.3rem; height: 1.3rem; }
.brand-name { font-size: 1.02rem; font-weight: 700; color: var(--ink); line-height: 1.2; }
.brand-sub { font-size: 0.7rem; color: var(--faint); margin-top: 0.1rem; }

[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background-color: var(--card) !important; border-color: var(--line) !important; border-radius: 8px !important;
}
[data-baseweb="input"] input, [data-baseweb="select"] input { color: var(--ink) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--blue) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--card) !important; border: 1px solid var(--line); }
div[data-testid="stNumberInput"] button { background: var(--card); border-color: var(--line); }

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) { gap: 0.8rem !important; margin-bottom: 0.9rem; }
div[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 0.9rem 1.05rem;
    transition: border-color .15s ease;
}
div[data-testid="stMetric"]:hover { border-color: #26384a; }
div[data-testid="stMetricLabel"] p { font-size: 0.78rem; font-weight: 500; color: var(--slate); }
div[data-testid="stMetricValue"] {
    font-size: 1.55rem; font-weight: 700; color: var(--ink); line-height: 1.25;
    font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
}
div[data-testid="stMetricDelta"] { font-size: 0.78rem; font-variant-numeric: tabular-nums; }

div[data-baseweb="tab-list"] { gap: 0.2rem; border-bottom: 1px solid var(--line); overflow-x: auto; }
button[data-baseweb="tab"] { padding: 0.65rem 0.95rem; background: transparent; border-radius: 8px 8px 0 0; white-space: nowrap; }
button[data-baseweb="tab"] p { font-size: 0.86rem; font-weight: 500; color: var(--slate); }
button[data-baseweb="tab"][aria-selected="true"] { background: rgba(22,133,255,0.10); }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); font-weight: 600; }
button[data-baseweb="tab"]:hover p { color: var(--ink); }
[data-baseweb="tab-highlight"] { background-color: var(--blue) !important; height: 2px !important; }
[data-baseweb="tab-border"] { background-color: transparent !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
div[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; margin-bottom: 0.8rem; }
details[data-testid="stExpander"] { border: none !important; }
details[data-testid="stExpander"] summary { padding: 0.6rem 0.9rem !important; }
details[data-testid="stExpander"] summary:hover { background: var(--card); }
details[data-testid="stExpander"] summary span p { font-weight: 500; font-size: 0.9rem; }
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stCaptionContainer"] { color: var(--slate); }
.stButton > button { border-radius: 8px; background: var(--card); border: 1px solid var(--line); color: var(--ink);
                     transition: border-color .15s ease, background .15s ease; }
.stButton > button:hover { border-color: var(--blue); background: #0f1a24; color: var(--ink); }
.stButton > button:active { background: #12202d; }
.stButton > button[kind="primary"] { background: var(--blue); border-color: var(--blue); font-weight: 600; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #2b93ff; border-color: #2b93ff; }
[data-testid="stPlotlyChart"] { position: relative; background: var(--panel); border-radius: 14px; padding: 0; overflow: hidden; }
[data-testid="stPlotlyChart"]::after { content: ""; position: absolute; inset: 0; border: 1px solid var(--line);
                                       border-radius: 14px; pointer-events: none; }
[data-testid="stPlotlyChart"] .main-svg { background: transparent !important; }
[data-testid="stPlotlyChart"] .main-svg .bg { fill: transparent !important; }
[data-testid="stSegmentedControl"] button { border-radius: 8px; }

.s-pill { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: var(--slate);
          background: var(--card); padding: 0.22rem 0.7rem; border-radius: 100px; border: 1px solid var(--line); }
.s-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.s-live { background: var(--green); }
.s-snap { background: var(--amber); }

.a-note { background: rgba(22,133,255,0.07); border-left: 3px solid var(--blue); border-radius: 0 10px 10px 0;
          padding: 0.9rem 1.15rem; font-size: 0.92rem; line-height: 1.7; color: var(--slate); max-width: none; margin: 0.5rem 0 1.1rem; }
.a-note p { margin: 0.25rem 0; }
.a-note b { color: var(--ink); }

.sec { font-size: 1.02rem; font-weight: 600; color: var(--ink); margin: 1.5rem 0 0.65rem; }

.sig { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.8rem; border-radius: 100px; font-size: 0.85rem; font-weight: 600; }
.sig-b { background: rgba(0,214,143,0.12); color: var(--green); border: 1px solid rgba(0,214,143,0.35); }
.sig-s { background: rgba(255,82,99,0.12); color: var(--red); border: 1px solid rgba(255,82,99,0.35); }

.t-foot { text-align: center; font-size: 0.78rem; color: var(--faint); padding: 0.9rem 0 0.3rem; border-top: 1px solid var(--line); margin-top: 2rem; }
h5 { margin-top: 0.8rem; }

.topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; }
.chip { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 0.35rem 0.8rem; line-height: 1.25; }
.chip-l { font-size: 0.68rem; color: var(--faint); }
.chip-v { font-size: 0.86rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.hero-t { font-size: 2.1rem; font-weight: 800; color: var(--ink); letter-spacing: -0.03em; line-height: 1.15; margin: 0.6rem 0 0.15rem; }
.hero-t span { color: var(--cyan); }
.hero-s { font-size: 0.98rem; color: var(--slate); margin-bottom: 1rem; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.8rem; margin: 0.4rem 0 1rem; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 0.95rem 1.05rem; transition: border-color .15s ease; }
.kpi:hover { border-color: #26384a; }
.kpi-top { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.35rem; }
.kpi-ico { width: 1.9rem; height: 1.9rem; border-radius: 8px; display: grid; place-items: center; flex: none; }
.kpi-ico svg { width: 1rem; height: 1rem; }
.kpi-l { font-size: 0.78rem; color: var(--slate); font-weight: 500; }
.kpi-v { font-size: 1.7rem; font-weight: 700; color: var(--ink); letter-spacing: -0.025em; font-variant-numeric: tabular-nums; line-height: 1.2; }
.kpi-v.na { color: var(--faint); }
.kpi-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 0.5rem; margin-top: 0.3rem; min-height: 2rem; }
.kpi-d { font-size: 0.78rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.kpi-d small { display: block; color: var(--faint); font-weight: 400; font-size: 0.7rem; }
.pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--slate); }
.kpi svg.spark { width: 46%; height: 2rem; flex: none; }

.pnl { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; height: 100%; }
.pnl-h { font-size: 1rem; font-weight: 600; color: var(--ink); }
.pnl-s { font-size: 0.78rem; color: var(--faint); margin-bottom: 0.7rem; }
.tbl-wrap { overflow-x: auto; }
table.hold { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
table.hold th { text-align: right; font-weight: 500; color: var(--faint); font-size: 0.74rem; padding: 0.5rem 0.6rem;
                background: var(--card); white-space: nowrap; }
table.hold th:first-child, table.hold td:first-child { text-align: left; }
table.hold td { text-align: right; padding: 0.55rem 0.6rem; color: var(--ink); border-bottom: 1px solid var(--line);
                font-variant-numeric: tabular-nums; white-space: nowrap; }
table.hold td:first-child { font-weight: 600; }
table.hold tr:last-child td { border-bottom: 0; }
table.hold tr:hover td { background: rgba(255,255,255,0.02); }

.rk { display: grid; grid-template-columns: minmax(7.5rem, 11rem) 1fr auto; align-items: center; gap: 0.75rem; padding: 0.5rem 0; border-bottom: 1px solid var(--line); }
.rk:last-child { border-bottom: 0; }
.rk-l { font-size: 0.84rem; color: var(--slate); }
.rk-bar { height: 8px; background: #101a23; border-radius: 100px; overflow: hidden; }
.rk-bar i { display: block; height: 100%; border-radius: 100px; min-width: 2px; }
.rk-v { font-size: 0.88rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; text-align: right; min-width: 4.2rem; }
.rk-note { font-size: 0.7rem; color: var(--faint); margin-top: 0.6rem; }

.empty { border: 1px dashed var(--line); border-radius: 12px; padding: 1.4rem; text-align: center; color: var(--faint); font-size: 0.88rem; }


.mi { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
      font-weight: normal; font-style: normal; font-size: 1.1em; line-height: 1; letter-spacing: normal;
      text-transform: none; display: inline-block; white-space: nowrap; vertical-align: -0.2em;
      font-feature-settings: 'liga'; -webkit-font-feature-settings: 'liga'; color: var(--cyan); margin-right: 0.3rem; }
table.ref { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
table.ref th { text-align: left; font-weight: 500; color: var(--faint); font-size: 0.74rem; padding: 0.55rem 0.75rem;
               background: var(--card); white-space: nowrap; }
table.ref td { text-align: left; vertical-align: top; padding: 0.6rem 0.75rem; color: var(--slate); border-bottom: 1px solid var(--line); line-height: 1.5; }
table.ref td:first-child { color: var(--ink); font-weight: 500; }
table.ref tr:hover td { background: rgba(255,255,255,0.02); }
.ref-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }

@media (max-width: 1100px) {
    .hero-t { font-size: 1.7rem; }
}
@media (max-width: 768px) {
    .hero-t { font-size: 1.4rem; }
    .kpi-v { font-size: 1.4rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    .rk { grid-template-columns: 1fr auto; }
    .rk-bar { grid-column: 1 / -1; order: 3; }
}
</style>
"""

_LOGO = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#37B9FF" stroke-width="2.2" stroke-linecap="round" '
    'stroke-linejoin="round"><polyline points="3,17 9,11 13,15 21,6"/><polyline points="15,6 21,6 21,12"/></svg>'
)


ICONS = {
    "bars": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="20" x2="6" y2="12"/><line x1="12" y1="20" x2="12" y2="5"/><line x1="18" y1="20" x2="18" y2="9"/></svg>',
    "percent": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="19" y1="5" x2="5" y2="19"/><circle cx="7" cy="7" r="2"/><circle cx="17" cy="17" r="2"/></svg>',
    "shield": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></svg>',
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/></svg>',
}


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def esc(s) -> str:
    return html.escape(str(s))


def brand_html() -> str:
    return (
        f'<div class="brand"><div class="brand-mark">{_LOGO}</div><div>'
        '<div class="brand-name">Portfolio Terminal</div>'
        '<div class="brand-sub">Data-Driven Investment Analytics</div></div></div>'
    )


def chip_html(label: str, value: str, delta: float | None = None) -> str:
    d = ""
    if _ok(delta):
        d = f' <span class="{"pos" if delta >= 0 else "neg"}">{delta:+.2%}</span>'
    return f'<div class="chip"><div class="chip-l">{esc(label)}</div><div class="chip-v">{esc(value)}{d}</div></div>'


def hero_html() -> str:
    return (
        '<div class="hero-t">Welcome to <span>Portfolio Terminal</span></div>'
        '<div class="hero-s">Professional portfolio analysis and optimization with data-driven insights</div>'
    )


def sparkline(values, color: str) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return ""
    if len(v) > 80:
        v = v[np.linspace(0, len(v) - 1, 80).astype(int)]
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i / (len(v) - 1) * 100:.1f},{28 - (x - lo) / span * 26 - 1:.1f}" for i, x in enumerate(v))
    return (f'<svg class="spark" viewBox="0 0 100 28" preserveAspectRatio="none"><polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="1.6" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>')


def kpi_card(label: str, value: str | None, icon: str, color: str, delta: str = "", delta_cls: str = "neu",
             delta_note: str = "", spark: str = "", tip: str = "") -> str:
    na = value is None
    val = "—" if na else esc(value)
    title = f' title="{esc(tip)}"' if tip else ""
    dnote = f"<small>{esc(delta_note)}</small>" if delta_note else ""
    drow = f'<div class="kpi-d {delta_cls}">{esc(delta)}{dnote}</div>' if (delta or delta_note) else "<div></div>"
    return (
        f'<div class="kpi"{title}><div class="kpi-top">'
        f'<div class="kpi-ico" style="background:{color}22;color:{color}">{ICONS[icon]}</div>'
        f'<div class="kpi-l">{esc(label)}</div></div>'
        f'<div class="kpi-v{" na" if na else ""}">{val}</div>'
        f'<div class="kpi-row">{drow}{spark}</div></div>'
    )


def kpi_grid(cards: list[str]) -> str:
    return f'<div class="kpis">{"".join(cards)}</div>'


def section_html(title: str) -> str:
    return f'<div class="sec">{esc(title)}</div>'


def delta_cls(x) -> str:
    return "neu" if not _ok(x) else ("pos" if x >= 0 else "neg")


def holdings_html(rows: list[dict]) -> str:
    if not rows:
        return '<div class="empty">Chưa có dữ liệu danh mục để hiển thị.</div>'

    def cell_pct(x, signed=True):
        if not _ok(x):
            return '<td class="neu">—</td>'
        return f'<td class="{"pos" if x >= 0 else "neg"}">{x:+.2%}</td>' if signed else f"<td>{x:.1%}</td>"

    body = ""
    for r in rows:
        price = f"{r['price']:,.2f}" if _ok(r.get("price")) else "—"
        body += (f"<tr><td>{esc(r['ticker'])}</td><td>{r['weight']:.1%}</td><td>{price}</td>"
                 f"{cell_pct(r.get('d1'))}{cell_pct(r.get('period'))}</tr>")
    return (
        '<div class="tbl-wrap"><table class="hold"><thead><tr><th>Mã</th><th>Trọng số</th><th>Giá đóng cửa*</th>'
        f"<th>1 ngày</th><th>Cả kỳ</th></tr></thead><tbody>{body}</tbody></table></div>"
        '<div class="rk-note">*Giá đóng cửa đã điều chỉnh (cổ tức, chia tách), đơn vị theo sàn niêm yết.</div>'
    )


def risk_rows_html(items: list[dict]) -> str:
    out = ""
    for it in items:
        bar = '<div class="rk-bar"></div>'
        if it.get("frac") is not None and _ok(it["frac"]):
            w = max(0.0, min(1.0, float(it["frac"]))) * 100
            bar = f'<div class="rk-bar"><i style="width:{w:.1f}%;background:{it.get("color", COLORS["blue"])}"></i></div>'
        out += (f'<div class="rk" title="{esc(it.get("tip", ""))}"><div class="rk-l">{esc(it["label"])}</div>'
                f'{bar}<div class="rk-v">{esc(it["text"])}</div></div>')
    return out


EMOJI_TO_MATERIAL = {
    "📊": "dashboard", "⚠️": "shield", "🧬": "functions", "🎯": "tune",
    "📡": "sensors", "🔁": "history", "🚀": "speed",
}


def with_icons(text: str) -> str:
    out = esc(text)
    for emo, name in EMOJI_TO_MATERIAL.items():
        out = out.replace(emo, f'<span class="mi" aria-hidden="true">{name}</span>')
    return out


def ref_table_html(columns: list[str], rows: list[tuple]) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{with_icons(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="ref-wrap"><table class="ref"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
