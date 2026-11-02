from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

SNAPSHOT_PATH = Path(__file__).parent / "data" / "prices_snapshot.csv"
CACHE_DIR = Path(__file__).parent / "data" / "cache"


PRESET_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK", "LLY",
    "ABBV", "KO", "PEP", "AVGO", "COST", "ORCL", "BAC", "CSCO", "ADBE", "CRM",
    "MCD", "TMO", "ACN", "ABT", "NFLX", "AMD", "DIS", "WFC", "INTC", "QCOM",
    "TXN", "CAT", "GS", "IBM", "AMGN", "HON", "LOW", "UPS", "NKE", "PFE",
]


def clean_tickers(raw: str) -> list[str]:
    out: list[str] = []
    for t in raw.replace(";", ",").replace(" ", ",").split(","):
        t = t.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def _download(tickers, start, end=None) -> pd.DataFrame:
    return yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _flatten_close(raw: pd.DataFrame, tickers) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        if "Close" not in cols.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(list(tickers)[0])
    else:
        if "Close" not in cols:
            return pd.DataFrame()
        close = raw[["Close"]].copy()
        close.columns = [list(tickers)[0]]
    close = close.copy()
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.dropna(axis=1, how="all").dropna(how="all")


def _read_snapshot() -> pd.DataFrame | None:
    try:
        if SNAPSHOT_PATH.exists():
            return pd.read_csv(SNAPSHOT_PATH, index_col=0, parse_dates=True)
    except Exception:
        pass
    return None


def _save_snapshot(close: pd.DataFrame) -> None:
    try:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        old = _read_snapshot()
        merged = close if old is None else close.combine_first(old)
        merged.sort_index().to_csv(SNAPSHOT_PATH)
    except Exception:
        pass


@st.cache_data(ttl=3600, show_spinner="Đang tải dữ liệu từ Yahoo Finance...")
def load_prices(symbols: tuple[str, ...], start: str) -> tuple[pd.DataFrame, str]:
    try:
        close = _flatten_close(_download(symbols, start), symbols)
    except Exception:
        close = pd.DataFrame()

    if not close.empty:
        _save_snapshot(close)
        return close, "yahoo"

    snap = _read_snapshot()
    if snap is not None:
        cols = [s for s in symbols if s in snap.columns]
        if cols:
            out = snap.loc[snap.index >= pd.Timestamp(start), cols].dropna(how="all")
            if not out.empty:
                return out, "snapshot"
    raise RuntimeError(
        "Không tải được dữ liệu từ Yahoo Finance và chưa có snapshot dự phòng."
    )


@st.cache_data(ttl=3600, show_spinner="Đang tải dữ liệu OHLC...")
def load_ohlc(ticker: str, start: str) -> pd.DataFrame:
    try:
        raw = _download([ticker], start)
    except Exception:
        return pd.DataFrame()
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[need].dropna(subset=["Close"])


def align_report(prices: pd.DataFrame, cols: list[str], start: pd.Timestamp) -> dict:
    sub = prices[cols]
    first_valid = sub.apply(lambda s: s.first_valid_index())
    common = sub.dropna()
    window = sub.loc[start:]
    lost = int(len(window) - len(window.dropna())) if len(window) else 0
    limiting = first_valid.idxmax() if len(first_valid) else None
    return {
        "limiting": limiting,
        "limiting_first": first_valid.get(limiting) if limiting else None,
        "lost_rows": lost,
        "total_rows": int(len(window)),
        "common_start": common.index[0] if len(common) else None,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500_symbols() -> tuple[list[str], str]:
    try:
        import io
        import urllib.request

        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (portfolio-terminal; educational)"},
        )
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        tbl = pd.read_html(io.StringIO(html))[0]
        syms = [str(s).strip().replace(".", "-") for s in tbl["Symbol"].tolist()]
        if len(syms) > 400:
            return syms, "wikipedia"
    except Exception:
        pass
    return list(PRESET_UNIVERSE), "preset"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def read_disk_cache(key: str, max_age_h: float = 24.0) -> pd.DataFrame | None:
    p = _cache_path(key)
    try:
        if p.exists() and (time.time() - p.stat().st_mtime) < max_age_h * 3600:
            return pd.read_parquet(p)
    except Exception:
        pass
    return None


def write_disk_cache(key: str, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_cache_path(key))
    except Exception:
        pass


def load_universe_timed(symbols: tuple[str, ...], start: str, chunk: int = 50,
                        use_disk_cache: bool = True) -> tuple[pd.DataFrame, float, int, str]:
    import hashlib

    key = hashlib.md5(("|".join(symbols) + start).encode()).hexdigest()[:16]
    t0 = time.perf_counter()
    if use_disk_cache:
        cached = read_disk_cache(key)
        if cached is not None and not cached.empty:
            return cached, time.perf_counter() - t0, 0, "parquet"

    frames, n_chunks = [], 0
    for i in range(0, len(symbols), chunk):
        part = symbols[i : i + chunk]
        n_chunks += 1
        try:
            frames.append(_flatten_close(_download(part, start), part))
        except Exception:
            continue
    frames = [f for f in frames if not f.empty]
    if frames:
        prices = pd.concat(frames, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        if use_disk_cache:
            write_disk_cache(key, prices)
        return prices, time.perf_counter() - t0, n_chunks, "yahoo"

    snap = _read_snapshot()
    if snap is not None:
        cols = [s for s in symbols if s in snap.columns]
        if cols:
            return snap.loc[snap.index >= pd.Timestamp(start), cols], time.perf_counter() - t0, 0, "snapshot"
    raise RuntimeError("Không tải được dữ liệu cho tập mã đã chọn.")
