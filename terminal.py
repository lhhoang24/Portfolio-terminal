from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import analytics as an
import charts as ch
import theme
from data import (PRESET_UNIVERSE, align_report, clean_tickers, load_ohlc, load_prices,
                  load_sp500_symbols, load_universe_timed)

MAX_ASSETS = 15
WARMUP_DAYS = 420

st.set_page_config(page_title="Portfolio Analytics Terminal", page_icon=":material/monitoring:", layout="wide")

SUMMARY_FMT = {
    "Tổng lợi suất": "{:.1%}", "Lợi suất năm hóa": "{:.1%}", "Biến động năm": "{:.1%}",
    "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max Drawdown": "{:.1%}",
    "Beta": "{:.2f}", "Alpha (năm)": "{:.1%}",
}


def _stop_with_ai_reset() -> None:
    st.session_state.pop("portfolio_ai_context", None)
    _stop_with_ai_reset()


st.markdown(theme.CSS, unsafe_allow_html=True)


from ai_chat_widget import render_floating_chat
render_floating_chat()


def show_df(df: pd.DataFrame, fmt: dict | None = None) -> None:
    styled = df.style.format(fmt, na_rep="–") if fmt else df.style.format(na_rep="–")
    st.dataframe(styled, width="stretch")


def pct(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.2%}"


def num(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"


def tip(text: str) -> None:
    st.caption(text)


def section(title: str) -> None:
    st.markdown(theme.section_html(title), unsafe_allow_html=True)


def sharpe_word(s: float) -> str:
    if not np.isfinite(s):
        return "chưa xác định"
    if s < 0:
        return "thấp hơn cả lãi suất phi rủi ro"
    if s < 0.5:
        return "thấp"
    if s < 1:
        return "trung bình"
    if s < 2:
        return "tốt"
    return "rất cao – nên kiểm tra kỹ vì kỳ quan sát ngắn có thể làm con số này đẹp hơn thực tế"


def analyst_note(port: pd.Series, bench: pd.Series, beta: float, alpha: float,
                 benchmark: str, rf: float) -> str:
    tp, tb = an.total_return(port), an.total_return(bench)
    diff = (tp - tb) * 100
    vol, mdd, sh = an.annual_vol(port), an.max_drawdown(port), an.sharpe(port, rf)
    p1 = (f"Trong kỳ này danh mục {'tăng' if tp >= 0 else 'giảm'} <b>{abs(tp):.1%}</b>, "
          f"{'vượt' if diff >= 0 else 'kém'} {benchmark} ({tb:.1%}) khoảng <b>{abs(diff):.1f} điểm %</b>.")
    p2 = (f"Để đạt kết quả đó, danh mục dao động khoảng {vol:.1%} mỗi năm và có lúc sụt "
          f"<b>{abs(mdd):.1%}</b> từ đỉnh. Chỉ số Sharpe là {sh:.2f}, thuộc mức {sharpe_word(sh)}.")
    p3 = ""
    if np.isfinite(beta):
        p3 = (f"Beta là {beta:.2f}: khi {benchmark} đổi 1%, danh mục thường đổi khoảng {abs(beta):.2f}% "
              f"{'cùng chiều' if beta >= 0 else 'ngược chiều'}.")
        if np.isfinite(alpha):
            p3 += f" Phần lợi suất không giải thích được bởi thị trường (alpha) khoảng {alpha:.1%} mỗi năm."
    p4 = "Đây là kết quả của quá khứ, không dự báo cho tương lai."
    return "".join(f"<p>{p}</p>" for p in (p1, p2, p3, p4) if p)


HELP = {
    "ret": "Tổng mức tăng/giảm của giá trị danh mục từ ngày bắt đầu đến phiên cuối.",
    "sharpe": "Lợi suất vượt lãi suất phi rủi ro trên mỗi đơn vị biến động. Càng cao càng tốt; trên 1 là khá.",
    "vol": "Độ lên xuống của danh mục trong một năm (độ lệch chuẩn của lợi suất ngày, năm hóa).",
    "mdd": "Mức sụt sâu nhất tính từ một đỉnh xuống đáy kế tiếp.",
    "beta": "Danh mục dao động mạnh gấp mấy lần benchmark. Beta 1,5: benchmark đổi 1% thì danh mục thường đổi ~1,5%.",
    "alpha": "Lợi suất năm phần vượt trội mà biến động của benchmark không giải thích được (theo CAPM).",
    "treynor": "Lợi suất vượt lãi suất phi rủi ro trên mỗi đơn vị beta.",
    "ir": "Lợi suất vượt benchmark chia cho mức chênh lệch biến động so với benchmark. Cao nghĩa là vượt ổn định.",
    "var": "Mức lỗ trong một ngày mà chỉ khoảng 5% số ngày tệ hơn (lịch sử).",
    "cvar": "Mức lỗ trung bình của những ngày tệ hơn ngưỡng VaR.",
}

PRESETS = {
    "Big Tech": "AAPL,MSFT,NVDA",
    "Đa ngành": "AAPL,JPM,XOM,JNJ,WMT",
    "Ngân hàng": "JPM,BAC,WFC,GS",
    "Bán dẫn": "NVDA,AMD,AVGO,QCOM,TXN",
    "Phòng thủ (đối chứng)": "JNJ,WMT,XOM,JPM",
}


def _apply_preset() -> None:
    choice = st.session_state.get("preset", "Tự nhập")
    if choice in PRESETS:
        st.session_state["tickers_raw"] = PRESETS[choice]


_W_EPS = 1e-9


def rebalance_weights(weights: dict, pinned: list, edited: str, new_value: float):
    w = {k: float(v) for k, v in weights.items()}
    pinned = [p for p in pinned if p in w]
    new_value = min(max(float(new_value), 0.0), 1.0)
    others_pinned = sum(w[p] for p in pinned if p != edited)
    max_allowed = max(0.0, 1.0 - others_pinned)
    free = [k for k in w if k not in pinned and k != edited]
    msg = None

    if not free:

        if abs(new_value - max_allowed) > 1e-6:
            msg = (f"Không đổi được {edited}: các mã còn lại đã được chỉnh nên {edited} phải là "
                   f"{max_allowed:.2%} để tổng đủ 100%. Hãy chọn 'Đều nhau' để đặt lại, "
                   f"hoặc chỉnh một mã khác.")
        w[edited] = max_allowed
        return w, pinned, msg

    if new_value > max_allowed + _W_EPS:
        msg = (f"{edited} tối đa {max_allowed:.2%} vì các mã đã chỉnh trước đó "
               f"({', '.join(p for p in pinned if p != edited)}) đang giữ {others_pinned:.2%}. "
               f"Đã giới hạn về {max_allowed:.2%}.")
        new_value = max_allowed

    w[edited] = new_value
    pinned = [p for p in pinned if p != edited] + [edited]
    remain = max(0.0, 1.0 - sum(w[p] for p in pinned))
    cur = sum(w[f] for f in free)
    for f in free:
        if remain <= _W_EPS:
            w[f] = 0.0
        elif cur > _W_EPS:
            w[f] = remain * w[f] / cur
        else:
            w[f] = remain / len(free)
    return {k: round(v, 10) for k, v in w.items()}, pinned, msg


def _weights_reset(assets: list[str]) -> None:
    for t in assets:
        st.session_state[f"w_{t}"] = 1.0 / len(assets)
    st.session_state["w_pinned"] = []
    st.session_state["w_assets"] = tuple(assets)


def _on_weight_change(ticker: str) -> None:
    assets_ = list(st.session_state["w_assets"])
    old = {t: st.session_state[f"w_{t}"] for t in assets_}
    new_val = old[ticker]

    others = {t: v for t, v in old.items() if t != ticker}
    prev_ticker = 1.0 - sum(others.values())
    base = {**others, ticker: max(prev_ticker, 0.0)}
    w, pinned, msg = rebalance_weights(base, st.session_state.get("w_pinned", []), ticker, new_val)
    for t in assets_:
        st.session_state[f"w_{t}"] = w[t]
    st.session_state["w_pinned"] = pinned
    if msg:
        st.session_state["w_msg"] = msg


st.markdown(theme.hero_html(), unsafe_allow_html=True)
st.caption("Chọn mã cổ phiếu ở thanh bên trái để phân tích hiệu suất, rủi ro và so sánh với benchmark.")
status_slot = st.empty()

with st.expander(":material/menu_book: Cách app hoạt động & Giải thích thuật ngữ",
                 expanded=False):
    st.markdown(
        """
**Cách app hoạt động**

1. **Bạn chọn** mã cổ phiếu, benchmark (mốc so sánh), ngày bắt đầu và trọng số ở thanh bên trái.
2. **App tải giá** từ Yahoo Finance một lần cho cả danh sách. App lấy dư khoảng 14 tháng trước ngày bắt đầu để các đường trung bình dài (như MA200) có đủ dữ liệu. Kết quả được lưu 1 giờ, nên đổi thông số không phải tải lại.
3. **App tính toán:** giá → lợi suất mỗi ngày → lợi suất của cả danh mục theo trọng số → các chỉ số như Sharpe, beta, drawdown.
4. **Các tab hiển thị kết quả.** Mỗi lần bạn đổi một ô, Streamlit chạy lại toàn bộ file `terminal.py` từ trên xuống và cập nhật mọi tab.

**Nên xem theo thứ tự:** Danh mục (kết luận) → Rủi ro → Tín hiệu → Tối ưu hóa → Backtest.
Giá & chỉ báo và CAPM & yếu tố là phần tra cứu thêm. Tab Quy mô lớn cho thấy app xử lý được nhiều mã cùng lúc.

**Thuật ngữ**

| Thuật ngữ | Nghĩa đơn giản |
|---|---|
| Benchmark | Mốc so sánh, thường là SPY (đại diện chỉ số S&P 500) |
| Sharpe | Lợi suất nhận được trên mỗi đơn vị rủi ro. Càng cao càng tốt |
| Beta | Danh mục dao động mạnh gấp mấy lần thị trường |
| Alpha | Phần lợi suất vượt trội không do thị trường kéo lên |
| Max Drawdown | Mức sụt sâu nhất từ đỉnh xuống đáy |
| VaR 95% | Mức lỗ trong một ngày mà chỉ khoảng 5% số ngày tệ hơn |
| Backtest | Thử một chiến lược trên dữ liệu quá khứ để xem nó sẽ cho kết quả gì |
"""
    )


st.session_state.setdefault("tickers_raw", "AAPL,MSFT,NVDA")
with st.sidebar:
    st.markdown(theme.brand_html(), unsafe_allow_html=True)
    st.markdown("### Tài sản")
    st.selectbox("Bộ mã mẫu", ["Tự nhập"] + list(PRESETS), key="preset", on_change=_apply_preset,
                 help="Chọn nhanh một nhóm mã. Bạn vẫn có thể sửa trong ô bên dưới.")
    tickers_raw = st.text_input("Mã cổ phiếu", key="tickers_raw",
                                help="Cách nhau bằng dấu phẩy, dùng mã Yahoo Finance. Ví dụ: AAPL, MSFT, BRK-B.")
    benchmark = (st.text_input("Benchmark", "SPY", help="Chỉ số hoặc ETF dùng làm mốc so sánh.")
                 .strip().upper()) or "SPY"

    st.markdown("### Phân tích")
    start = st.date_input("Ngày bắt đầu phân tích", value=date(2023, 1, 1),
                          min_value=date(2010, 1, 1), max_value=date.today())
    rf = st.number_input("Lãi suất phi rủi ro (%/năm)", 0.0, 20.0, 4.0, 0.25,
                         help="Lãi suất của khoản đầu tư gần như không rủi ro (như trái phiếu chính phủ). "
                              "Dùng để tính Sharpe, alpha và Treynor.") / 100.0
    window = st.slider("Cửa sổ rolling (phiên)", 10, 120, 20,
                       help="Số phiên dùng cho các đường 'cuộn' như beta cuộn và biến động cuộn. "
                            "Ngắn thì nhạy nhưng nhiễu, dài thì mượt nhưng chậm.")

tickers = clean_tickers(tickers_raw)
if not tickers:
    st.warning("Hãy nhập ít nhất một mã cổ phiếu.")
    _stop_with_ai_reset()
if len(tickers) > MAX_ASSETS:
    st.warning(f"Chỉ dùng {MAX_ASSETS} mã đầu tiên cho danh mục (tab 'Quy mô' dành cho số lượng lớn).")
    tickers = tickers[:MAX_ASSETS]

start_ts = pd.Timestamp(start)
warm_start = start_ts - pd.Timedelta(days=WARMUP_DAYS)
symbols = tuple(dict.fromkeys(tickers + [benchmark]))


try:
    prices_all, source = load_prices(symbols, warm_start.date().isoformat())
except Exception as exc:
    st.error(f"Không tải được dữ liệu: {exc}")
    st.info("Mẹo: kiểm tra mã cổ phiếu/kết nối mạng, hoặc dùng file snapshot dự phòng trong thư mục data/.")
    _stop_with_ai_reset()
missing = [s for s in symbols if s not in prices_all.columns]
if benchmark in missing:
    st.error(f"Không có dữ liệu cho benchmark '{benchmark}'.")
    _stop_with_ai_reset()
assets = [t for t in tickers if t in prices_all.columns]
if missing:
    st.warning("Không có dữ liệu cho: " + ", ".join(missing) + " (đã bỏ qua).")
if not assets:
    st.error("Không có mã nào hợp lệ.")
    _stop_with_ai_reset()
all_cols = list(dict.fromkeys(assets + [benchmark]))
common = prices_all[all_cols].dropna()
full_rets = common.pct_change().dropna()
rep_al = align_report(prices_all, all_cols, start_ts)
if rep_al["lost_rows"] > 0:
    lim = rep_al["limiting"]
    lim_d = rep_al["limiting_first"]
    st.info(
        f"**Căn chỉnh dữ liệu:** app chỉ giữ những ngày mà MỌI mã đều có giá. Trong khoảng đã chọn, "
        f"{rep_al['lost_rows']}/{rep_al['total_rows']} phiên bị loại. Mã có lịch sử ngắn nhất là "
        f"**{lim}** (có giá từ {pd.Timestamp(lim_d).date() if lim_d is not None else '?'}). "
        "Mã niêm yết muộn hoặc khác lịch giao dịch (mã Việt Nam, crypto) sẽ làm mẫu co lại."
    )
if source != "yahoo" and missing:
    st.error(
        "Đang ở chế độ snapshot và snapshot **chưa có** các mã: " + ", ".join(missing) +
        ". Chạy `python scripts/build_snapshot.py` trên máy có mạng rồi commit file snapshot mới."
    )

rets = full_rets.loc[start_ts:]
if len(rets) < 60:
    st.error("Quá ít dữ liệu trong khoảng đã chọn (cần ≥ 60 phiên). Hãy lùi ngày bắt đầu.")
    _stop_with_ai_reset()

with st.sidebar:
    st.markdown("### Trọng số")
    wmode = st.radio("Cách chia trọng số", ["Đều nhau", "Tùy chỉnh"], horizontal=True,
                     label_visibility="collapsed")
    custom_w = wmode == "Tùy chỉnh" and len(assets) > 1

    if not custom_w:
        st.session_state["w_assets"] = None
        st.session_state["w_pinned"] = []
    elif st.session_state.get("w_assets") != tuple(assets):
        _weights_reset(assets)

    if custom_w:
        for t in assets:
            st.slider(t, 0.0, 1.0, step=0.01, key=f"w_{t}", on_change=_on_weight_change, args=(t,))
        weights = an.normalize_weights(pd.Series({t: st.session_state[f"w_{t}"] for t in assets}))
        _wmsg = st.session_state.pop("w_msg", None)
        if _wmsg:
            st.warning(_wmsg)
    else:
        weights = pd.Series(1.0 / len(assets), index=assets)
    st.caption("Tổng 100%: " + " · ".join(f"{t} {w:.2%}" for t, w in weights.items()))

    st.markdown("### Dữ liệu")
    if source == "yahoo":
        st.markdown(
            f'<div class="s-pill"><span class="s-dot s-live"></span>Yahoo Finance · '
            f'phiên cuối {prices_all.index[-1].date()}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="s-pill"><span class="s-dot s-snap"></span>Snapshot dự phòng · '
            f'phiên cuối {prices_all.index[-1].date()}</div>',
            unsafe_allow_html=True,
        )
    if st.button("Làm mới dữ liệu", help="Xóa cache giá (1 giờ) và tải lại từ nguồn.", width="stretch"):
        load_prices.clear()
        st.rerun()


factor_prices, factor_note, market_regime = None, None, None
try:
    _fp, _fsrc = load_prices(tuple(an.FACTOR_TICKERS), warm_start.date().isoformat())
    _fmiss = [f for f in an.FACTOR_TICKERS if f not in _fp.columns]
    if _fmiss:
        factor_note = ("Thiếu dữ liệu yếu tố: " + ", ".join(_fmiss) + ". Nếu đang ở chế độ snapshot, chạy "
                       "`python scripts/build_snapshot.py` trên máy có mạng để bổ sung QQQ, IWM, ^VIX.")
    else:
        factor_prices = _fp[list(an.FACTOR_TICKERS)].dropna()
        _ron = an.risk_on_off(factor_prices)["risk_on"].dropna()
        if len(_ron):
            market_regime = "RISK ON" if _ron.iloc[-1] == 1.0 else "RISK OFF"
except Exception as exc:
    factor_note = f"Không tải được dữ liệu yếu tố (SPY, QQQ, IWM, ^VIX): {exc}"

bench_full = full_rets[benchmark]
bench = rets[benchmark]
port_full = an.portfolio_returns(full_rets, weights)
port = port_full.loc[start_ts:]
_bn = common[benchmark]


def _ai_safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

_ai_beta, _ai_alpha, _ai_r2 = an.beta_alpha(port, bench, rf)
_ai_var95, _ai_cvar95 = an.var_cvar(port, 0.95)
_ai_holdings = []
for _ticker in weights.index:
    _px = common[_ticker]
    _ai_holdings.append({
        "ticker": str(_ticker),
        "weight": _ai_safe_float(weights[_ticker]),
        "last_price": _ai_safe_float(_px.iloc[-1]) if len(_px) else None,
        "period_return": _ai_safe_float(an.total_return(rets[_ticker])),
        "daily_return": _ai_safe_float(rets[_ticker].iloc[-1]) if len(rets[_ticker]) else None,
    })
_ai_context = {
    "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    "data_source": "Yahoo Finance" if source == "yahoo" else "Snapshot dự phòng",
    "period": f"{start_ts.date()} đến {prices_all.index[-1].date()} ({len(rets)} phiên)",
    "benchmark": benchmark,
    "tickers": list(map(str, assets)),
    "weights": {str(k): _ai_safe_float(v) for k, v in weights.items()},
    "total_return": _ai_safe_float(an.total_return(port)),
    "annual_return": _ai_safe_float(an.annual_return(port)),
    "annual_volatility": _ai_safe_float(an.annual_vol(port)),
    "sharpe": _ai_safe_float(an.sharpe(port, rf)),
    "sortino": _ai_safe_float(an.sortino(port, rf)),
    "max_drawdown": _ai_safe_float(an.max_drawdown(port)),
    "beta": _ai_safe_float(_ai_beta),
    "alpha_annual": _ai_safe_float(_ai_alpha),
    "r_squared": _ai_safe_float(_ai_r2),
    "information_ratio": _ai_safe_float(an.information_ratio(port, bench)),
    "treynor": _ai_safe_float(an.treynor(port, _ai_beta, rf)),
    "var_95_daily": _ai_safe_float(_ai_var95),
    "cvar_95_daily": _ai_safe_float(_ai_cvar95),
    "last_portfolio_daily_return": _ai_safe_float(port.iloc[-1]) if len(port) else None,
    "last_benchmark_daily_return": _ai_safe_float(bench.iloc[-1]) if len(bench) else None,
    "risk_free_rate_annual": _ai_safe_float(rf),
    "market_regime": market_regime,
    "holdings": _ai_holdings,
}
st.session_state["portfolio_ai_context"] = _ai_context
_b1d = float(_bn.iloc[-1] / _bn.iloc[-2] - 1) if len(_bn) >= 2 else None
_p1d = float(port.iloc[-1]) if len(port) else None
_chips = [
    theme.chip_html(benchmark, f"{float(_bn.iloc[-1]):,.2f}", _b1d),
    theme.chip_html("Danh mục · 1 ngày", "—" if _p1d is None else f"{_p1d:+.2%}"),
    theme.chip_html("Dữ liệu", f"{len(rets)} phiên · {len(assets)} mã"),
    theme.chip_html("Phiên cuối", str(prices_all.index[-1].date())),
    theme.chip_html("Nguồn", "Yahoo Finance" if source == "yahoo" else "Snapshot dự phòng"),
]
if market_regime:
    _chips.insert(3, theme.chip_html("Chế độ thị trường", market_regime))
status_slot.markdown(f'<div class="topbar">{"".join(_chips)}</div>', unsafe_allow_html=True)
if source != "yahoo":
    st.warning("Yahoo Finance không phản hồi – ứng dụng đang dùng dữ liệu snapshot đã lưu trước đó.")


tabs = st.tabs([
    ":material/dashboard: Danh mục", ":material/candlestick_chart: Giá & chỉ báo", ":material/shield: Rủi ro",
    ":material/sensors: Tín hiệu", ":material/tune: Tối ưu hóa", ":material/history: Backtest",
    ":material/functions: CAPM & yếu tố", ":material/speed: Quy mô lớn", ":material/info: Ghi chú",
    ":material/menu_book: Đối chiếu bài báo",
])


with tabs[0]:
    beta, alpha, r2 = an.beta_alpha(port, bench, rf)
    var95_d, cvar95_d = an.var_cvar(port, 0.95)
    tip("Danh mục này lời hay lỗ, và so với thị trường thì thế nào?")


    def _fin(x) -> bool:
        return x is not None and bool(np.isfinite(x))

    port_cum = an.cumulative(port)
    growth = float(port_cum.iloc[-1]) if len(port_cum) else float("nan")
    tr_p, tr_b = an.total_return(port), an.total_return(bench)
    ar_p, ar_b = an.annual_return(port), an.annual_return(bench)
    vol_p, vol_b = an.annual_vol(port), an.annual_vol(bench)
    sh_p, sh_b = an.sharpe(port, rf), an.sharpe(bench, rf)
    vol_roll = an.rolling_vol(port_full, window).loc[start_ts:]
    G, R, B, A = theme.COLORS["green"], theme.COLORS["red"], theme.COLORS["blue"], theme.COLORS["amber"]
    P = theme.COLORS["purple"]

    st.markdown(theme.kpi_grid([
        theme.kpi_card("Giá trị của 1 đơn vị đầu tư", f"{growth:.3f}" if _fin(growth) else None, "bars", G,
                       f"{tr_p:+.1%} cả kỳ" if _fin(tr_p) else "", theme.delta_cls(tr_p),
                       f"{benchmark} {tr_b:+.1%}" if _fin(tr_b) else "",
                       theme.sparkline(port_cum.values, G), HELP["ret"]),
        theme.kpi_card("Lợi suất năm hóa", pct(ar_p) if _fin(ar_p) else None, "percent", P,
                       f"{(ar_p - ar_b) * 100:+.1f} điểm %" if _fin(ar_p) and _fin(ar_b) else "",
                       theme.delta_cls(ar_p - ar_b if _fin(ar_p) and _fin(ar_b) else None),
                       f"so với {benchmark}", "", "Lợi suất bình quân mỗi năm trong kỳ phân tích."),
        theme.kpi_card("Rủi ro (biến động năm)", pct(vol_p) if _fin(vol_p) else None, "shield", R,
                       f"{(vol_p - vol_b) * 100:+.1f} điểm %" if _fin(vol_p) and _fin(vol_b) else "",
                       theme.delta_cls(-(vol_p - vol_b) if _fin(vol_p) and _fin(vol_b) else None),
                       f"so với {benchmark}", theme.sparkline(vol_roll.values, R), HELP["vol"]),
        theme.kpi_card("Sharpe", num(sh_p) if _fin(sh_p) else None, "target", A,
                       f"{sh_p - sh_b:+.2f}" if _fin(sh_p) and _fin(sh_b) else "",
                       theme.delta_cls(sh_p - sh_b if _fin(sh_p) and _fin(sh_b) else None),
                       f"so với {benchmark}", "", HELP["sharpe"]),
    ]), unsafe_allow_html=True)

    st.markdown(f'<div class="a-note">{analyst_note(port, bench, beta, alpha, benchmark, rf)}</div>',
                unsafe_allow_html=True)


    section("Hiệu suất")
    c = st.columns(4)
    c[0].metric("Lợi suất danh mục", pct(an.total_return(port)), help=HELP["ret"])
    c[1].metric(f"Lợi suất {benchmark}", pct(an.total_return(bench)))
    diff_val = an.total_return(port) - an.total_return(bench)
    c[2].metric("Chênh lệch vs benchmark",
                f"{diff_val * 100:+.1f} điểm %",
                delta=f"{'vượt' if diff_val >= 0 else 'kém'} benchmark",
                delta_color="normal" if diff_val >= 0 else "inverse")
    c[3].metric("Sharpe", num(an.sharpe(port, rf)), help=HELP["sharpe"])

    section("Rủi ro & thị trường")
    c = st.columns(6)
    c[0].metric("Biến động năm", pct(an.annual_vol(port)), help=HELP["vol"])
    c[1].metric("Max Drawdown", pct(an.max_drawdown(port)), help=HELP["mdd"])
    c[2].metric(f"Beta vs {benchmark}", num(beta), help=HELP["beta"])
    c[3].metric("Alpha CAPM (năm)", pct(alpha), help=HELP["alpha"])
    c[4].metric("Information ratio", num(an.information_ratio(port, bench)), help=HELP["ir"])
    c[5].metric("Treynor", num(an.treynor(port, beta, rf)), help=HELP["treynor"])


    left, right = st.columns([2, 1])
    with left:
        ranges = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
                  "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5)}

        avail = [k for k, off in ranges.items() if port.index[-1] - off > port.index[0]] + ["Toàn kỳ"]
        if st.session_state.get("perf_range") not in avail:
            st.session_state["perf_range"] = "Toàn kỳ"
        pick = st.segmented_control("Khoảng thời gian", avail, key="perf_range",
                                    label_visibility="collapsed") or "Toàn kỳ"
        if pick in ranges:
            cut = port.index[-1] - ranges[pick]
            p_v, b_v = port.loc[port.index > cut], bench.loc[bench.index > cut]
        else:
            p_v, b_v = port, bench
        growth_v = pd.DataFrame({"Danh mục": an.cumulative(p_v), benchmark: an.cumulative(b_v)})
        st.plotly_chart(ch.perf_fig(growth_v, f"Hiệu suất tích lũy · {pick}"), width="stretch")
    with right:
        st.plotly_chart(ch.pie_fig(weights, "Phân bổ trọng số", f"{int((weights > 1e-6).sum())} mã"),
                        width="stretch")

    with st.expander("Biểu đồ gốc: giá trị của 1 đơn vị đầu tư", expanded=False):
        growth = pd.DataFrame({"Danh mục": an.cumulative(port), benchmark: an.cumulative(bench)})
        st.plotly_chart(ch.line_fig(growth, "Giá trị của 1 đơn vị đầu tư", hline=1.0), width="stretch")


    h_rows = []
    for t in weights[assets].sort_values(ascending=False).index[:8]:
        px_ = common[t]
        h_rows.append({
            "ticker": t, "weight": float(weights[t]),
            "price": float(px_.iloc[-1]) if len(px_) else None,
            "d1": float(px_.iloc[-1] / px_.iloc[-2] - 1) if len(px_) >= 2 else None,
            "period": an.total_return(rets[t]),
        })
    hl, hr = st.columns(2)
    hl.markdown(
        '<div class="pnl"><div class="pnl-h">Top Holdings</div><div class="pnl-s">Các vị thế lớn nhất theo trọng số</div>'
        + theme.holdings_html(h_rows) + "</div>", unsafe_allow_html=True)

    mdd_p = an.max_drawdown(port)
    risk_items = [
        {"label": "Biến động năm", "text": pct(vol_p), "tip": HELP["vol"], "frac": vol_p, "color": R},
        {"label": "Max Drawdown", "text": pct(mdd_p), "tip": HELP["mdd"], "frac": abs(mdd_p) if _fin(mdd_p) else None, "color": A},
        {"label": "VaR 95% (1 ngày)", "text": pct(var95_d), "tip": HELP["var"], "frac": abs(var95_d) if _fin(var95_d) else None, "color": P},
        {"label": "CVaR 95% (1 ngày)", "text": pct(cvar95_d), "tip": HELP["cvar"], "frac": abs(cvar95_d) if _fin(cvar95_d) else None, "color": P},
        {"label": f"Beta vs {benchmark}", "text": num(beta), "tip": HELP["beta"]},
        {"label": "Alpha CAPM (năm)", "text": pct(alpha), "tip": HELP["alpha"]},
        {"label": "Sortino", "text": num(an.sortino(port, rf)), "tip": "Giống Sharpe nhưng chỉ tính biến động phía lỗ."},
        {"label": "Information ratio", "text": num(an.information_ratio(port, bench)), "tip": HELP["ir"]},
    ]
    hr.markdown(
        '<div class="pnl"><div class="pnl-h">Risk Metrics</div><div class="pnl-s">Chỉ số rủi ro của danh mục</div>'
        + theme.risk_rows_html(risk_items)
        + '<div class="rk-note">Độ dài thanh = độ lớn trên thang 0–100% (không chuẩn hóa); các chỉ số tỷ lệ không có thanh.</div></div>',
        unsafe_allow_html=True)

    section("Từng tài sản")
    per_asset = {a: rets[a] for a in assets}
    per_asset["Danh mục"] = port
    show_df(an.summary_table(per_asset, bench, rf), SUMMARY_FMT)


with tabs[1]:
    tip("Giá của từng mã đã đi như thế nào, và đường trung bình đang nằm trên hay dưới giá?")
    c = st.columns([1, 1, 1, 1])
    sel = c[0].selectbox("Mã", all_cols)
    kind = c[1].radio("Loại biểu đồ", ["Candlestick", "Line"], horizontal=True)
    sma_w = c[2].number_input("SMA (phiên)", 2, 250, 20)
    ema_w = c[3].number_input("EMA (phiên)", 2, 250, 20)
    show_ind = st.multiselect("Hiển thị", ["SMA", "EMA"], default=["SMA", "EMA"])

    ohlc = load_ohlc(sel, warm_start.date().isoformat())
    if ohlc.empty:
        st.warning("Không tải được OHLC – hiển thị đường giá đóng cửa.")
        ohlc = pd.DataFrame({"Close": common[sel]})
        for col in ("Open", "High", "Low"):
            ohlc[col] = ohlc["Close"]
        ohlc["Volume"] = np.nan
        kind = "Line"
    overlays = {}
    if "SMA" in show_ind:
        overlays[f"SMA {sma_w}"] = ohlc["Close"].rolling(int(sma_w)).mean().loc[start_ts:]
    if "EMA" in show_ind:
        overlays[f"EMA {ema_w}"] = ohlc["Close"].ewm(span=int(ema_w), adjust=False).mean().loc[start_ts:]
    view = ohlc.loc[start_ts:]
    if len(view) >= 2:
        last, prev = float(view["Close"].iloc[-1]), float(view["Close"].iloc[-2])
        section("Tóm tắt")
        m = st.columns(4)
        m[0].metric("Giá đóng cửa gần nhất", f"{last:,.2f}",
                    delta=f"{(last / prev - 1) * 100:+.2f}%",
                    delta_color="normal" if last >= prev else "inverse")
        m[1].metric("Cao nhất kỳ", f"{float(view['High'].max()):,.2f}")
        m[2].metric("Thấp nhất kỳ", f"{float(view['Low'].min()):,.2f}")
        vol_avg = view["Volume"].mean()
        m[3].metric("KL giao dịch TB", "–" if not np.isfinite(vol_avg) else f"{vol_avg:,.0f}")
        st.plotly_chart(ch.candlestick_fig(view, sel, overlays, kind), width="stretch")
        with st.expander("Bảng dữ liệu"):
            show_df(view.tail(60).round(2))
    else:
        st.info("Chưa đủ dữ liệu để hiển thị biểu đồ. Hãy chọn khoảng thời gian dài hơn.")


with tabs[2]:
    tip("Các mã có đi cùng nhau không, danh mục từng sụt sâu bao nhiêu, và một ngày xấu có thể lỗ bao nhiêu?")

    section("Tương quan giữa các tài sản")
    if len(assets) > 1:
        st.plotly_chart(ch.corr_heatmap(rets[assets].corr()), width="stretch")
    else:
        st.info("Cần ≥ 2 mã để vẽ ma trận tương quan.")

    section("Drawdown")
    dd = pd.DataFrame({"Danh mục": an.drawdown(port), benchmark: an.drawdown(bench)})
    st.plotly_chart(ch.line_fig(dd, "So sánh drawdown", pct=True), width="stretch")

    section("Biến động & Beta cuộn")
    vol_df = pd.DataFrame({"Danh mục": an.rolling_vol(port_full, window),
                           benchmark: an.rolling_vol(bench_full, window)}).loc[start_ts:]
    st.plotly_chart(ch.line_fig(vol_df, f"Biến động cuộn {window} phiên (năm hóa)", pct=True),
                    width="stretch")

    rb = an.rolling_beta(port_full, bench_full, window).loc[start_ts:]
    st.plotly_chart(ch.line_fig(rb.rename("Beta"), f"Beta cuộn {window} phiên so với {benchmark}",
                                hline=1.0), width="stretch")

    section("Tiếp xúc momentum (theo bài báo)")
    mom_exp = an.rolling_momentum_exposure(port_full, common[benchmark], window).loc[start_ts:]
    if len(mom_exp):
        st.plotly_chart(ch.line_fig(mom_exp.rename("Tương quan"),
                                    f"Tiếp xúc momentum cuộn {window} phiên (tương quan với momentum 12 phiên của {benchmark})",
                                    hline=0.0), width="stretch")
        st.caption(f"Momentum = P(t)/P(t−12) − 1 tính trên giá {benchmark}, như bài báo. Vì lấy từ chính benchmark nên "
                   "chỉ số này gần với tương quan thị trường hơn là một yếu tố độc lập; momentum 12-1 riêng từng mã "
                   "nằm ở tab 'CAPM & yếu tố'.")
    else:
        st.info("Chưa đủ dữ liệu để tính tiếp xúc momentum cuộn.")

    section("VaR & phân phối lợi suất")
    var95, cvar95 = an.var_cvar(port, 0.95)
    c = st.columns(3)
    c[0].metric("VaR 95% (1 ngày)", pct(var95), help=HELP["var"])
    c[1].metric("CVaR 95% (1 ngày)", pct(cvar95), help=HELP["cvar"])
    c[2].metric("Ngày xấu nhất", pct(float(port.min())))
    st.plotly_chart(ch.hist_fig(port, "Phân phối lợi suất ngày của danh mục", var95),
                    width="stretch")
    st.caption("VaR/CVaR lịch sử: mức lỗ trong ngày bị vượt với xác suất ~5%, và mức lỗ trung bình khi vượt ngưỡng đó.")


with tabs[3]:
    tip("Theo quy tắc đơn giản 'đường trung bình ngắn nằm trên đường dài', mỗi mã đang ở trạng thái mua hay bán?")
    c = st.columns(2)
    s_win = c[0].slider("MA ngắn", 3, 50, 20, key="sig_s")
    l_win = c[1].slider("MA dài", 10, 250, 50, key="sig_l")
    if s_win >= l_win:
        st.warning("MA ngắn nên nhỏ hơn MA dài.")
    sig_full = an.ma_signals(common[assets], s_win, l_win)
    sig = sig_full.loc[start_ts:]
    share_full = (sig_full * weights[assets]).sum(axis=1, skipna=False).dropna()
    share = share_full.loc[start_ts:]
    if len(share):
        last_share = float(share.iloc[-1])
        is_buy = last_share > 0.5
        sig_label = "BUY" if is_buy else "SELL"
        sig_cls = "sig-b" if is_buy else "sig-s"
        sig_icon = '<span class="mi" aria-hidden="true">%s</span>' % ("trending_up" if is_buy else "trending_down")
        st.markdown(
            f'<div class="sig {sig_cls}">{sig_icon} {sig_label} — '
            f'{last_share:.0%} trọng số có MA ngắn &gt; MA dài</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(ch.line_fig(share.rename("Tỷ trọng tín hiệu BUY"),
                                    "Tỷ trọng danh mục có tín hiệu BUY", pct=True),
                        width="stretch")

    section("Chi tiết từng mã")
    ma_s_last = common[assets].rolling(s_win).mean().iloc[-1]
    ma_l_last = common[assets].rolling(l_win).mean().iloc[-1]
    table = pd.DataFrame({
        "Giá": common[assets].iloc[-1], f"MA{s_win}": ma_s_last, f"MA{l_win}": ma_l_last,
        "Tín hiệu": np.where(ma_s_last > ma_l_last, "BUY", "SELL"),
    })
    show_df(table, {"Giá": "{:,.2f}", f"MA{s_win}": "{:,.2f}", f"MA{l_win}": "{:,.2f}"})
    with st.expander("20 phiên gần nhất"):
        show_df(sig.tail(20).apply(lambda col: col.map({1.0: "BUY", 0.0: "SELL"})))
    st.caption("Chỉ mang tính minh họa: tín hiệu tính trên giá đóng cửa phiên cuối, không phải khuyến nghị đầu tư.")


with tabs[4]:
    tip("Nếu dựa vào dữ liệu quá khứ, nên chia tiền thế nào để lợi suất trên mỗi đơn vị rủi ro cao nhất? "
        "Kết quả này có còn đúng với giai đoạn chưa thấy không?")
    if len(assets) < 2:
        st.info("Cần ≥ 2 mã để tối ưu hóa.")
    else:
        n_a = len(assets)
        min_w = min(1.0, round(1.0 / n_a + 0.01, 2))
        max_w = st.slider("Trần trọng số mỗi mã", min_w, 1.0, max(min_w, min(1.0, 0.6)), 0.05)
        rets_a = rets[assets]
        w_sh = an.optimize_portfolio(rets_a, rf, max_w, "sharpe")
        w_mv = an.optimize_portfolio(rets_a, rf, max_w, "minvar")
        mu = rets_a.mean().values * an.TRADING_DAYS
        cov = rets_a.cov().values * an.TRADING_DAYS

        cands = {"Hiện tại": weights[assets], "Max Sharpe": w_sh, "Min Variance": w_mv}
        stats = {}
        for name, w in cands.items():
            r_, v_, s_ = an.portfolio_stats(w.values, mu, cov, rf)
            stats[name] = {"Lợi suất kỳ vọng": r_, "Biến động": v_, "Sharpe": s_}

        section("So sánh danh mục")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trọng số**")
            show_df(pd.DataFrame(cands), {k: "{:.1%}" for k in cands})
        with c2:
            st.markdown("**Chỉ số trong mẫu (in-sample)**")
            show_df(pd.DataFrame(stats).T, {"Lợi suất kỳ vọng": "{:.1%}", "Biến động": "{:.1%}", "Sharpe": "{:.2f}"})

        pc = st.columns(2)
        with pc[0]:
            st.plotly_chart(ch.pie_fig(w_sh, "Phân bổ tối ưu (Max Sharpe)"), width="stretch")
        with pc[1]:
            st.plotly_chart(ch.pie_fig(weights[assets], "Phân bổ hiện tại"), width="stretch")

        section("Đường biên hiệu quả (chỉ gồm danh mục thỏa trần trọng số)")
        fr = an.monte_carlo_frontier(rets_a, rf, 3000, max_weight=max_w)
        asset_df = pd.DataFrame({"Lợi suất": mu, "Biến động": np.sqrt(np.diag(cov))}, index=assets)
        pts = {n_: (s_["Biến động"], s_["Lợi suất kỳ vọng"]) for n_, s_ in stats.items()}
        st.plotly_chart(ch.frontier_fig(fr, asset_df, {
            "Max Sharpe": pts["Max Sharpe"], "Min Variance": pts["Min Variance"],
            "Danh mục hiện tại": pts["Hiện tại"]}), width="stretch")

        section("Kiểm tra ngoài mẫu (out-of-sample)")
        frac = st.slider("Tỷ lệ dữ liệu dùng để huấn luyện", 0.5, 0.9, 0.7, 0.05)
        split = int(len(rets_a) * frac)
        train, test = rets_a.iloc[:split], rets_a.iloc[split:]
        if len(train) >= 60 and len(test) >= 20:
            w_tr = an.optimize_portfolio(train, rf, max_w, "sharpe")
            cand_r = {
                "Max Sharpe (tối ưu trên train)": an.portfolio_returns(test, w_tr),
                "Đều nhau": an.portfolio_returns(test, pd.Series(1.0 / n_a, index=assets)),
                "Danh mục hiện tại": an.portfolio_returns(test, weights[assets]),
            }
            st.caption(f"Train: {train.index[0].date()} → {train.index[-1].date()} · "
                       f"Test: {test.index[0].date()} → {test.index[-1].date()}")
            st.plotly_chart(ch.line_fig(pd.DataFrame({k: an.cumulative(v) for k, v in cand_r.items()}),
                                        "Hiệu suất trên tập test", hline=1.0), width="stretch")
            show_df(an.summary_table(cand_r, rets.loc[test.index, benchmark], rf), SUMMARY_FMT)
            st.caption("Trọng số 'tối ưu' dựa trên lợi suất trung bình lịch sử rất nhiễu, nên thường kém ổn định "
                       "ngoài mẫu – đó là lý do cần kiểm tra out-of-sample.")
        else:
            st.info("Không đủ dữ liệu để chia train/test – hãy lùi ngày bắt đầu.")


with tabs[5]:
    tip("Nếu đã áp dụng từng chiến lược trong quá khứ thì kết quả sẽ ra sao, kể cả chi phí giao dịch?")

    section("Tham số chiến lược")
    c = st.columns(4)
    bs = c[0].slider("MA ngắn", 3, 50, 20, key="bt_s")
    bl = c[1].slider("MA dài", 10, 250, 50, key="bt_l")
    cost = c[2].slider("Chi phí giao dịch (bps)", 0, 50, 5)
    neutral = c[3].slider("Tỷ lệ cổ phiếu khi Neutral", 0.0, 1.0, 0.5, 0.1)

    bt_ma = an.backtest_ma_crossover(full_rets, common, weights, bs, bl, cost, rf).loc[start_ts:]
    reg = an.regime_score(common[benchmark])
    bt_reg_full, exposure_full = an.backtest_regime(port_full, reg["regime"], neutral, cost, rf)
    bt_reg = bt_reg_full.loc[start_ts:]
    bh = an.buy_hold_returns(full_rets, weights).loc[start_ts:]
    strategies = {
        "Buy & Hold (trọng số trôi)": bh,
        "Tái cân bằng hằng ngày": port,
        f"MA crossover {bs}/{bl}": bt_ma,
        "Regime switching": bt_reg,
        f"Benchmark ({benchmark})": bench,
    }

    section("So sánh chiến lược")
    st.plotly_chart(ch.line_fig(pd.DataFrame({k: an.cumulative(v) for k, v in strategies.items()}),
                                "Giá trị của 1 đơn vị đầu tư", hline=1.0), width="stretch")
    show_df(an.summary_table(strategies, bench, rf), SUMMARY_FMT)

    section("Drawdown")
    st.plotly_chart(ch.line_fig(pd.DataFrame({k: an.drawdown(v) for k, v in strategies.items()}),
                                "Drawdown các chiến lược", pct=True), width="stretch")

    section("Regime thị trường")
    reg_view = reg.loc[start_ts:]
    if reg_view["regime"].isna().any():
        st.warning(f"{int(reg_view['regime'].isna().sum())} phiên chưa đủ dữ liệu để xác định regime "
                   "(chiến lược regime giữ tiền mặt trong các phiên đó).")
    valid_reg = reg_view.dropna(subset=["score"])
    if len(valid_reg):
        st.plotly_chart(ch.regime_fig(valid_reg["score"]), width="stretch")
        st.plotly_chart(ch.regime_price_map(common[benchmark].reindex(valid_reg.index), valid_reg["regime"],
                                            f"{benchmark} – bản đồ regime (Bull / Neutral / Bear)"),
                        width="stretch")
        dist = valid_reg["regime"].value_counts(normalize=True).reindex(["Bull", "Neutral", "Bear"]).fillna(0)
        cc = st.columns(3)
        for col, name in zip(cc, ["Bull", "Neutral", "Bear"]):
            col.metric(f"% thời gian {name}", f"{dist[name]:.0%}")
        with st.expander("10 phiên gần nhất"):
            show_df(valid_reg.tail(10))
    st.caption("**Buy & Hold** mua một lần theo trọng số ban đầu và không bán/mua thêm, nên mã tăng mạnh dần "
               "chiếm tỷ trọng lớn. **Tái cân bằng hằng ngày** luôn kéo về trọng số cố định, nên khác Buy & Hold "
               "khi các mã tăng không đều.")
    st.caption("Tín hiệu luôn được trễ 1 phiên (quyết định hôm nay áp dụng cho lợi suất phiên sau) "
               "để tránh look-ahead bias. Chi phí giao dịch tính theo mức thay đổi vị thế.")


with tabs[6]:
    tip(f"Mỗi mã nhạy với {benchmark} đến đâu, có tạo ra phần lợi suất vượt trội không, và mã nào đang có đà tăng mạnh?")

    section("Bảng CAPM & Momentum")
    rows = {}
    mom = an.momentum_12_1(common[assets])
    for a in assets:
        b_, al_, r2_ = an.beta_alpha(rets[a], bench, rf)
        rows[a] = {"Beta": b_, "Alpha (năm)": al_, "R²": r2_,
                   "Treynor": an.treynor(rets[a], b_, rf), "Momentum 12-1": mom.get(a, np.nan)}
    capm_df = pd.DataFrame(rows).T
    capm_df["Hạng momentum"] = capm_df["Momentum 12-1"].rank(ascending=False)
    show_df(capm_df, {"Beta": "{:.2f}", "Alpha (năm)": "{:.1%}", "R²": "{:.2f}", "Treynor": "{:.2f}",
                      "Momentum 12-1": "{:.1%}", "Hạng momentum": "{:.0f}"})
    st.caption("Momentum 12-1 = lợi suất từ t−252 đến t−21 phiên (bỏ tháng gần nhất). "
               "Cần ≥ 253 phiên lịch sử, nếu không sẽ hiện '–'.")

    section("Beta cuộn")
    rb_all = an.rolling_beta(full_rets[assets], bench_full, window).loc[start_ts:]
    st.plotly_chart(ch.beta_heatmap(rb_all, f"Beta cuộn {window} phiên so với {benchmark}"),
                    width="stretch")

    hi_b, lo_b = an.highest_lowest_beta(rb_all, 3)
    if len(assets) >= 2 and len(hi_b):
        section("Mã có beta cao nhất / thấp nhất")
        cl, cr = st.columns(2)
        with cl:
            st.markdown("**Beta cao nhất**")
            show_df(hi_b.rename("Beta").to_frame(), {"Beta": "{:.2f}"})
        with cr:
            st.markdown("**Beta thấp nhất**")
            show_df(lo_b.rename("Beta").to_frame(), {"Beta": "{:.2f}"})
        st.caption(f"Beta cuộn {window} phiên mới nhất so với {benchmark}.")

    section("Đóng góp & lợi suất vượt trội")
    contrib = (rets[assets] * weights[assets]).sum()
    st.plotly_chart(ch.bar_fig(contrib, "Đóng góp vào lợi suất (tổng lợi suất ngày có trọng số)"),
                    width="stretch")
    active = (port - bench).cumsum()
    st.plotly_chart(ch.line_fig(active.rename("Lợi suất vượt trội cộng dồn"),
                                f"Lợi suất vượt trội so với {benchmark} (cộng dồn)",
                                pct=True, hline=0.0), width="stretch")

    section("Beta theo yếu tố thị trường (SPY · QQQ · IWM · VIX)")
    if factor_prices is None:
        st.info(factor_note or "Chưa có dữ liệu yếu tố.")
    else:
        f_rets = factor_prices.pct_change().dropna()
        fb = an.factor_betas(port, f_rets.loc[start_ts:])
        short = {"^VIX": "VIX"}
        fc = st.columns(4)
        for col, f in zip(fc, an.FACTOR_TICKERS):
            col.metric(f"Beta {short.get(f, f)}", num(float(fb.loc[f, "Beta"])))
        st.plotly_chart(ch.factor_beta_fig(fb["Beta"].rename(index=short), "Beta của danh mục theo yếu tố"),
                        width="stretch")
        fs = fb.copy()
        fs.insert(0, "Yếu tố", [an.FACTOR_NAMES[f] for f in fs.index])
        fs.index = [short.get(f, f) for f in fs.index]
        show_df(fs, {"Beta": "{:.2f}", "R²": "{:.2f}"})
        st.caption("Beta đơn biến: cov(danh mục, yếu tố) / var(yếu tố), tính riêng từng yếu tố như bài báo (không phải hồi quy "
                   "đa biến). Beta VIX thường âm: danh mục giảm khi biến động thị trường tăng.")

    section("Chế độ thị trường: Risk-On / Risk-Off")
    if factor_prices is None:
        st.info(factor_note or "Chưa có dữ liệu yếu tố.")
    else:
        ron = an.risk_on_off(factor_prices)
        ron_ok = ron.dropna(subset=["risk_on"])
        if ron_ok.empty:
            st.info("Chưa đủ dữ liệu (cần ≥ 200 phiên để tính MA200). Hãy lùi ngày bắt đầu.")
        else:
            last = ron_ok.iloc[-1]
            rc = st.columns(4)
            rc[0].metric("Chế độ hiện tại", market_regime or "–")
            rc[1].metric("SPY so với MA200", "Trên" if last["SPY > MA"] == 1.0 else "Dưới")
            rc[2].metric("QQQ so với MA200", "Trên" if last["QQQ > MA"] == 1.0 else "Dưới")
            rc[3].metric("VIX", f"{float(factor_prices['^VIX'].iloc[-1]):.1f}",
                         help="Risk-On cần VIX dưới 20.")
            share_on = float(ron_ok.loc[start_ts:, "risk_on"].mean()) if len(ron_ok.loc[start_ts:]) else float("nan")
            st.caption(f"Quy tắc (theo bài báo): RISK ON khi SPY và QQQ cùng trên MA200 **và** VIX < 20, "
                       f"ngược lại RISK OFF. Trong khoảng đã chọn, Risk-On chiếm {pct(share_on)} số phiên. "
                       "Đây là quy tắc chọn tay, chỉ mang tính minh họa.")
            with st.expander("10 phiên gần nhất"):
                show_df(ron.tail(10).apply(lambda col: col.map({1.0: "Có", 0.0: "Không"})))


with tabs[7]:
    tip("App xử lý được bao nhiêu mã cùng lúc? Tải theo lô, cache Parquet, so sánh tính vectorized với vòng lặp "
        "và xem thời gian tăng thế nào khi số mã tăng.")
    sp_syms, sp_src = load_sp500_symbols()
    universe = sp_syms if sp_src == "wikipedia" else list(PRESET_UNIVERSE)
    st.caption(f"Tập mã: **{len(universe)}** mã ({'S&P 500 từ Wikipedia' if sp_src == 'wikipedia' else 'danh sách mẫu 50 mã – không lấy được S&P 500'}).")

    cA, cB = st.columns([3, 1])
    n_u = cA.slider("Số mã cần quét", 10, len(universe), min(100, len(universe)), 10)
    use_cache = cB.checkbox("Cache Parquet", value=True,
                            help="Lưu giá đã tải xuống ổ đĩa (24 giờ). Bỏ chọn để đo thời gian tải thật từ Yahoo.")
    b1, b2 = st.columns(2)
    run_scan = b1.button("Chạy quét", type="primary", icon=":material/play_arrow:")
    run_scale = b2.button("Đo mức tăng theo số mã", icon=":material/trending_up:")

    if run_scan:
        syms = tuple(dict.fromkeys(universe[:n_u] + [benchmark]))
        try:
            with st.spinner("Đang tải và tính toán..."):
                prices_u, dl_s, n_chunks, src_u = load_universe_timed(syms, start_ts.date().isoformat(),
                                                                       use_disk_cache=use_cache)
                r_u = prices_u.loc[start_ts:].pct_change().dropna(how="all")
                bench_u = r_u[benchmark] if benchmark in r_u.columns else bench
                r_assets = r_u.drop(columns=[benchmark], errors="ignore")
                t0 = time.perf_counter()
                table_u = an.metrics_table(r_assets, bench_u, rf)
                t_vec = time.perf_counter() - t0
                t0 = time.perf_counter()
                an.metrics_table_loop(r_assets, bench_u, rf)
                t_loop = time.perf_counter() - t0
            st.session_state["uni"] = {
                "table": table_u, "dl": dl_s, "vec": t_vec, "loop": t_loop, "chunks": n_chunks, "src": src_u,
                "n_assets": table_u.shape[0], "rows": r_u.shape[0], "cells": int(r_u.notna().sum().sum()),
            }
        except Exception as exc:
            st.error(f"Không chạy được quét: {exc}")

    if run_scale:
        sizes = [n for n in (10, 25, 50, 100, 200, 500) if n <= len(universe)]
        recs = []
        prog = st.progress(0.0, text="Đang đo...")
        try:
            all_syms = tuple(dict.fromkeys(universe[: max(sizes)] + [benchmark]))
            prices_big, dl_s, _, _src = load_universe_timed(all_syms, start_ts.date().isoformat(),
                                                             use_disk_cache=use_cache)
            r_big = prices_big.loc[start_ts:].pct_change().dropna(how="all")
            b_big = r_big[benchmark] if benchmark in r_big.columns else bench
            for k, n in enumerate(sizes):
                cols = [c for c in r_big.columns if c != benchmark][:n]
                t0 = time.perf_counter(); an.metrics_table(r_big[cols], b_big, rf); tv = time.perf_counter() - t0
                t0 = time.perf_counter(); an.metrics_table_loop(r_big[cols], b_big, rf); tl = time.perf_counter() - t0
                recs.append({"Số mã": len(cols), "Vectorized (ms)": tv * 1000, "Vòng lặp (ms)": tl * 1000,
                             "Nhanh gấp": tl / tv if tv > 0 else np.nan})
                prog.progress((k + 1) / len(sizes), text=f"Đã đo {len(cols)} mã")
            st.session_state["scale"] = {"df": pd.DataFrame(recs).set_index("Số mã"), "dl": dl_s}
        except Exception as exc:
            st.error(f"Không đo được: {exc}")
        prog.empty()

    uni = st.session_state.get("uni")
    if uni:
        section("Kết quả quét")
        m = st.columns(5)
        m[0].metric("Số mã", uni["n_assets"])
        m[1].metric("Số phiên", f"{uni['rows']:,}")
        m[2].metric("Số quan sát", f"{uni['cells']:,}")
        src_txt = {"yahoo": f"{uni['chunks']} lô từ Yahoo", "parquet": "từ cache Parquet", "snapshot": "từ snapshot"}[uni["src"]]
        m[3].metric("Thời gian tải", f"{uni['dl']:.1f}s", src_txt)
        m[4].metric("Tính (vectorized)", f"{uni['vec'] * 1000:.0f} ms",
                    f"vòng lặp {uni['loop'] * 1000:.0f} ms", delta_color="off")
        if uni["src"] == "snapshot":
            st.warning("Yahoo không phản hồi: kết quả chỉ gồm các mã có trong snapshot.")
        st.caption("Thời gian tải chủ yếu là chờ mạng; phần tính toán thường nhỏ hơn nhiều. "
                   "Tải lại với cache Parquet sẽ gần như tức thì.")
        st.plotly_chart(ch.risk_return_scatter(uni["table"], "Rủi ro – lợi suất của toàn bộ tập mã"),
                        width="stretch")
        section("Top 10 theo Sharpe")
        fmt_u = {"Lợi suất năm hóa": "{:.1%}", "Biến động năm": "{:.1%}", "Sharpe": "{:.2f}",
                 "Max Drawdown": "{:.1%}", "Beta": "{:.2f}", "Số phiên": "{:.0f}"}
        show_df(uni["table"].head(10), fmt_u)
        with st.expander("Toàn bộ bảng"):
            show_df(uni["table"], fmt_u)

    sc = st.session_state.get("scale")
    if sc:
        section("Thời gian tính theo số mã")
        st.plotly_chart(ch.line_fig(sc["df"][["Vectorized (ms)", "Vòng lặp (ms)"]],
                                    "Thời gian tính bảng chỉ số (ms)"), width="stretch")
        show_df(sc["df"], {"Vectorized (ms)": "{:.1f}", "Vòng lặp (ms)": "{:.1f}", "Nhanh gấp": "{:.1f}×"})
    if not uni and not sc:
        st.info("Nhấn **Chạy quét** hoặc **Đo mức tăng theo số mã** ở trên để bắt đầu.")


with tabs[8]:
    tip("Dữ liệu và công thức app dùng, những gì app chưa làm được, và nguồn tham khảo.")
    st.markdown(
        """
**Dữ liệu & phương pháp**
- Giá đóng cửa đã điều chỉnh (cổ tức, chia tách) từ Yahoo Finance qua `yfinance`; cache 1 giờ.
- Lợi suất đơn giản theo ngày; danh mục chính tái cân bằng hằng ngày về trọng số cố định (tab Backtest có thêm Buy & Hold thật); năm hóa với 252 phiên.
- Chỉ dùng những ngày mà mọi mã đều có giá (inner join); app báo mã nào làm co mẫu.
- Sharpe/Sortino/CAPM dùng lãi suất phi rủi ro nhập ở thanh bên.
- Tối ưu hóa: Markowitz long-only (SLSQP) với trần trọng số; kỳ vọng lợi suất = trung bình lịch sử.
- Regime = xu hướng (MA50 > MA200) + động lượng (63 phiên > 0) + biến động thấp (dưới trung vị 1 năm) của benchmark.
- Backtest: tín hiệu trễ 1 phiên, có tính chi phí giao dịch; phần không đầu tư hưởng lãi suất phi rủi ro.

**Hạn chế**
- Lợi suất kỳ vọng từ trung bình lịch sử rất nhiễu → danh mục tối ưu thiếu ổn định ngoài mẫu.
- Chưa mô hình hóa trượt giá, thuế, thanh khoản; mô hình yếu tố (CAPM, momentum) còn đơn giản.
- Thiên lệch sống sót (chỉ dùng mã hiện đang niêm yết) và kết quả quá khứ không đảm bảo tương lai.
- Yahoo Finance là nguồn không chính thức, có thể giới hạn tần suất; app có snapshot dự phòng.

**Tham khảo ý tưởng**: bài viết "Building an Awesome Portfolio Management App in Streamlit from Scratch" (Alexzap, Medium, 2026)
và repo *Real-Time-Stock-Analysis-Dashboard-using-Streamlit* (AarushiBose, GitHub). Mã nguồn ở đây được viết lại và mở rộng.

**Tuyên bố**: chỉ phục vụ học tập, không phải lời khuyên đầu tư.
"""
    )


with tabs[9]:
    tip("App này đi theo bài báo của Alexzap đến đâu, và chủ động khác ở chỗ nào?")
    section("Đối chiếu từng phần")
    _ref_rows = [
            ("Portfolio (lợi suất, alpha, Sharpe, phân bổ)", "📊 Danh mục", "Thêm rf vào Sharpe/alpha; thêm Max Drawdown, Information ratio; bảng từng tài sản."),
            ("Risk (tương quan, drawdown, vol cuộn)", "⚠️ Rủi ro", "Thêm VaR/CVaR lịch sử và phân phối lợi suất; cửa sổ rolling chỉnh được."),
            ("Rolling beta vs SPY", "⚠️ Rủi ro, 🧬 CAPM", "Beta cuộn cho từng mã (heatmap), không chỉ cho danh mục."),
            ("Momentum exposure (tổng hợp từ SPY)", "⚠️ Rủi ro, 🧬 CAPM & yếu tố", "Có đường tương quan cuộn với momentum 12 phiên của SPY như bài (kèm cảnh báo: gần như đo tương quan thị trường); thêm Momentum 12-1 riêng từng mã."),
            ("Factors: beta SPY / QQQ / IWM / VIX, Factor Summary", "🧬 CAPM & yếu tố", "Beta đơn biến theo 4 yếu tố, biểu đồ cột và bảng tóm tắt kèm R²."),
            ("Risk-On / Risk-Off (Market Regime)", "🧬 CAPM & yếu tố, thanh trạng thái", "SPY và QQQ trên MA200 và VIX < 20; hiện cả điều kiện thành phần và 10 phiên gần nhất."),
            ("Highest / Lowest Beta Names", "🧬 CAPM & yếu tố", "Hai bảng mã có beta cuộn cao nhất / thấp nhất."),
            ("SPY Price Regime Map", "🔁 Backtest", "Giá benchmark tô màu theo Bull / Neutral / Bear."),
            ("Optimized Allocation (biểu đồ tròn)", "🎯 Tối ưu hóa", "Biểu đồ tròn Max Sharpe cạnh phân bổ hiện tại."),
            ("Max Sharpe optimization", "🎯 Tối ưu hóa", "Thêm Min Variance, trần trọng số, đường biên Monte Carlo (có tôn trọng trần) và kiểm tra out-of-sample."),
            ("Signals (MA crossover)", "📡 Tín hiệu", "Tín hiệu theo từng mã và tỷ trọng BUY của cả danh mục; MA chỉnh được."),
            ("Bull/Bear regime: Bull = Max Sharpe, Neutral = chia đều, Bear = tiền mặt", "🔁 Backtest",
             "KHÁC BÀI. Bài đổi trọng số theo regime; app giữ nguyên danh mục của bạn ở Bull, giữ tỷ lệ cổ phiếu chọn được ở "
             "Neutral (thanh trượt) và 0% ở Bear (phần còn lại hưởng lãi suất phi rủi ro), tín hiệu trễ 1 phiên, có chi phí giao dịch. "
             "Lý do: trọng số Max Sharpe của bài tối ưu trên toàn bộ mẫu rồi áp ngược vào quá khứ (nhìn trước), và giữ cùng danh mục "
             "để tách riêng hiệu ứng chọn thời điểm khỏi hiệu ứng đổi trọng số. Ngưỡng regime cũng khác: điểm 0–3 gồm xu hướng MA50 > MA200, "
             "động lượng 63 phiên (bài: 12 phiên) và biến động thấp hơn trung vị 1 năm (bài: dưới phân vị 70%); thêm trạng thái Neutral."),
            ("Treynor Ratio & CAPM Alpha (cấp danh mục)", "📊 Danh mục",
             "Alpha CAPM (năm) và Treynor hiện ở hàng 'Rủi ro & thị trường'. Bài dùng lãi suất phi rủi ro 0%, app mặc định 4% nên số khác bài; "
             "đặt về 0% ở thanh bên để đối chiếu."),
            ("Backtest vs SPY", "🔁 Backtest", "Tín hiệu trễ 1 phiên, có chi phí giao dịch, Buy & Hold thật vs tái cân bằng ngày."),
            ("Một file ~800 dòng", "app / analytics / charts / data", "Tách lớp; hàm phân tích thuần, không phụ thuộc Streamlit."),
            ("yf.download gọi lặp", "data.py", "Tải một lần, cache 1 giờ, snapshot dự phòng, cache Parquet cho tập lớn."),
            ("(Không có)", "🚀 Quy mô lớn", "Tải theo lô, vectorized vs vòng lặp, đường thời gian theo số mã."),
    ]
    st.markdown(theme.ref_table_html(["Bài báo", "Trong app", "Khác biệt / cải tiến"], _ref_rows),
                unsafe_allow_html=True)
    section("Những điểm bài báo tự nhận là hạn chế – app xử lý tới đâu")
    show_df(pd.DataFrame(
        [
            ("Lợi suất kỳ vọng từ trung bình lịch sử rất nhiễu", "Chưa giải quyết", "Chỉ cảnh báo và kiểm tra out-of-sample; hướng cải tiến: shrinkage, Black–Litterman."),
            ("Tối ưu Sharpe không ổn định", "Giảm bớt", "Trần trọng số + out-of-sample + so với danh mục đều nhau."),
            ("Regime theo luật, không ước lượng thống kê", "Chưa giải quyết", "Vẫn là luật chọn tay; hướng cải tiến: Hidden Markov, phân tích độ nhạy ngưỡng."),
            ("Bỏ qua chi phí giao dịch", "Đã xử lý một phần", "Có chi phí theo bps; chưa có trượt giá và thuế."),
            ("Gọi Yahoo lặp lại", "Đã xử lý", "Cache + snapshot + Parquet."),
            ("Mọi thứ trong một file", "Đã xử lý", "Tách 4 module."),
        ],
        columns=["Hạn chế nêu trong bài", "Trạng thái", "Ghi chú"],
    ).set_index("Hạn chế nêu trong bài"))
    st.caption("Lưu ý khi trình bày: bộ mã mặc định (AAPL, MSFT, NVDA) là nhóm thắng của giai đoạn này nên kết quả "
               "đẹp một phần do chọn sau khi đã biết kết quả; hãy đối chiếu với bộ 'Phòng thủ (đối chứng)'.")


st.markdown(
    '<div class="t-foot">'
    'Portfolio Analytics Terminal · Phục vụ học tập · Không phải lời khuyên đầu tư'
    '</div>',
    unsafe_allow_html=True,
)
