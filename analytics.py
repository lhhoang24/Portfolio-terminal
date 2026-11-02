from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def normalize_weights(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / s


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = list(weights.index)
    return (returns[cols] * weights[cols]).sum(axis=1)


def buy_hold_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = list(weights.index)
    value = (1.0 + returns[cols].fillna(0.0)).cumprod() * weights[cols]
    total = value.sum(axis=1)
    out = total.pct_change()
    out.iloc[0] = total.iloc[0] - 1.0
    return out


def cumulative(r):
    return (1.0 + r).cumprod()


def drawdown(r):
    c = cumulative(r)
    return c / c.cummax() - 1.0


def total_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def annual_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    growth = (1.0 + r).prod()
    return float(growth ** (TRADING_DAYS / len(r)) - 1.0) if growth > 0 else np.nan


def annual_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    r = r.dropna()
    sd = r.std()
    if len(r) < 2 or not np.isfinite(sd) or sd == 0:
        return 0.0
    return float((r.mean() - rf / TRADING_DAYS) / sd * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    r = r.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r.dropna()).min()) if len(r.dropna()) else np.nan


def var_cvar(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan
    q = float(r.quantile(1.0 - level))
    tail = r[r <= q]
    return q, float(tail.mean()) if len(tail) else q


def beta_alpha(port: pd.Series, bench: pd.Series, rf: float = 0.0) -> tuple[float, float, float]:
    df = pd.concat([port, bench], axis=1).dropna()
    if len(df) < 3:
        return np.nan, np.nan, np.nan
    rf_d = rf / TRADING_DAYS
    ep, eb = df.iloc[:, 0] - rf_d, df.iloc[:, 1] - rf_d
    var_b = eb.var()
    if var_b == 0:
        return np.nan, np.nan, np.nan
    beta = float(ep.cov(eb) / var_b)
    alpha = float((ep.mean() - beta * eb.mean()) * TRADING_DAYS)
    r2 = float(ep.corr(eb) ** 2)
    return beta, alpha, r2


def treynor(r: pd.Series, beta: float, rf: float = 0.0) -> float:
    if not np.isfinite(beta) or beta == 0:
        return np.nan
    return float((annual_return(r) - rf) / beta)


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    active = (r - bench).dropna()
    te = active.std() * np.sqrt(TRADING_DAYS)
    if len(active) < 2 or not np.isfinite(te) or te == 0:
        return np.nan
    return float(active.mean() * TRADING_DAYS / te)


def rolling_beta(r, bench: pd.Series, window: int):
    cov = r.rolling(window).cov(bench)
    var = bench.rolling(window).var()
    if isinstance(cov, pd.DataFrame):
        return cov.div(var, axis=0)
    return cov / var


def rolling_vol(r, window: int):
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)


def summary_table(returns: dict[str, pd.Series], bench: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    rows = {}
    for name, r in returns.items():
        beta, alpha, _ = beta_alpha(r, bench, rf)
        rows[name] = {
            "Tổng lợi suất": total_return(r),
            "Lợi suất năm hóa": annual_return(r),
            "Biến động năm": annual_vol(r),
            "Sharpe": sharpe(r, rf),
            "Sortino": sortino(r, rf),
            "Max Drawdown": max_drawdown(r),
            "Beta": beta,
            "Alpha (năm)": alpha,
        }
    return pd.DataFrame(rows).T


def metrics_table(returns: pd.DataFrame, bench: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    n = returns.notna().sum()
    growth = (1.0 + returns.fillna(0.0)).prod()
    ann_ret = growth ** (TRADING_DAYS / n.clip(lower=1)) - 1.0
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sh = ((returns.mean() - rf / TRADING_DAYS) / returns.std().replace(0, np.nan)) * np.sqrt(TRADING_DAYS)
    cum = (1.0 + returns.fillna(0.0)).cumprod()
    mdd = (cum / cum.cummax() - 1.0).min()
    beta = returns.corrwith(bench) * returns.std() / bench.std()
    out = pd.DataFrame(
        {
            "Lợi suất năm hóa": ann_ret,
            "Biến động năm": vol,
            "Sharpe": sh,
            "Max Drawdown": mdd,
            "Beta": beta,
            "Số phiên": n,
        }
    )
    return out.sort_values("Sharpe", ascending=False)


def momentum_12_1(prices: pd.DataFrame, skip: int = 21, lookback: int = 252) -> pd.Series:
    if len(prices) <= lookback:
        return pd.Series(np.nan, index=prices.columns)
    return prices.iloc[-1 - skip] / prices.iloc[-1 - lookback] - 1.0


def ma_signals(prices: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    ma_s = prices.rolling(short).mean()
    ma_l = prices.rolling(long).mean()
    sig = (ma_s > ma_l).astype(float)
    return sig.where(ma_l.notna())


def regime_score(bench_prices: pd.Series, fast: int = 50, slow: int = 200,
                 mom_window: int = 63, vol_window: int = 20) -> pd.DataFrame:
    ret = bench_prices.pct_change()
    ma_f = bench_prices.rolling(fast).mean()
    ma_s = bench_prices.rolling(slow).mean()
    mom = bench_prices.pct_change(mom_window)
    vol = ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    vol_med = vol.rolling(252, min_periods=60).median()

    trend = (ma_f > ma_s).astype(int)
    momentum = (mom > 0).astype(int)
    calm = (vol < vol_med).astype(int)
    score = trend + momentum + calm
    valid = ma_s.notna() & mom.notna() & vol_med.notna()
    score = score.where(valid)

    label = pd.Series(np.nan, index=score.index, dtype=object)
    label[score <= 0] = "Bear"
    label[score == 1] = "Neutral"
    label[score >= 2] = "Bull"
    label[score.isna()] = np.nan
    return pd.DataFrame({"score": score, "regime": label,
                         "trend": trend.where(valid), "momentum": momentum.where(valid),
                         "calm": calm.where(valid)})


def backtest_ma_crossover(returns: pd.DataFrame, prices: pd.DataFrame, weights: pd.Series,
                          short: int, long: int, cost_bps: float, rf: float) -> pd.Series:
    cols = list(weights.index)
    sig = ma_signals(prices[cols], short, long).reindex(returns.index).fillna(0.0)
    held = sig.shift(1).fillna(0.0)
    rf_d = rf / TRADING_DAYS
    gross = ((held * returns[cols] + (1.0 - held) * rf_d) * weights[cols]).sum(axis=1)
    turnover = (held.diff().abs().fillna(0.0) * weights[cols]).sum(axis=1)
    return gross - turnover * cost_bps / 1e4


def backtest_regime(port_ret: pd.Series, regime: pd.Series, neutral_exposure: float,
                    cost_bps: float, rf: float) -> tuple[pd.Series, pd.Series]:
    exposure = regime.map({"Bull": 1.0, "Neutral": neutral_exposure, "Bear": 0.0})
    exposure = exposure.reindex(port_ret.index).astype(float)
    held = exposure.shift(1).fillna(0.0)
    rf_d = rf / TRADING_DAYS
    net = held * port_ret + (1.0 - held) * rf_d - held.diff().abs().fillna(0.0) * cost_bps / 1e4
    return net, held


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol, (ret - rf) / vol if vol > 0 else 0.0


def optimize_portfolio(returns: pd.DataFrame, rf: float = 0.0, max_weight: float = 1.0,
                       objective: str = "sharpe") -> pd.Series:
    n = returns.shape[1]
    if n == 1:
        return pd.Series([1.0], index=returns.columns)
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    max_weight = max(max_weight, 1.0 / n + 1e-9)

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(w @ mu - rf) / vol

    def variance(w):
        return w @ cov @ w

    fun = neg_sharpe if objective == "sharpe" else variance
    res = minimize(
        fun, np.ones(n) / n, method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = pd.Series(res.x if res.success else np.ones(n) / n, index=returns.columns)
    return normalize_weights(w)


def monte_carlo_frontier(returns: pd.DataFrame, rf: float = 0.0, n: int = 3000,
                         seed: int = 42, max_weight: float = 1.0) -> pd.DataFrame:
    k = returns.shape[1]
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    rng = np.random.default_rng(seed)
    max_weight = max(max_weight, 1.0 / k + 1e-9)
    kept: list[np.ndarray] = []
    total = 0
    for _ in range(20):
        W = rng.dirichlet(np.ones(k), size=n)
        W = W[(W <= max_weight).all(axis=1)]
        kept.append(W)
        total += len(W)
        if total >= n:
            break
    W = np.vstack(kept)[:n] if total else np.ones((1, k)) / k
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    sh = np.where(vols > 0, (rets - rf) / vols, 0.0)
    return pd.DataFrame({"Lợi suất": rets, "Biến động": vols, "Sharpe": sh})


def metrics_table_loop(returns: pd.DataFrame, bench: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    rows = {}
    for c in returns.columns:
        r = returns[c].dropna()
        beta, _, _ = beta_alpha(r, bench, rf)
        rows[c] = {"Lợi suất năm hóa": annual_return(r), "Biến động năm": annual_vol(r),
                   "Sharpe": sharpe(r, rf), "Max Drawdown": max_drawdown(r), "Beta": beta,
                   "Số phiên": len(r)}
    return pd.DataFrame(rows).T


FACTOR_TICKERS = ("SPY", "QQQ", "IWM", "^VIX")
FACTOR_NAMES = {"SPY": "SPY (thị trường)", "QQQ": "QQQ (công nghệ)", "IWM": "IWM (vốn hóa nhỏ)",
                "^VIX": "VIX (biến động)"}


def factor_betas(port: pd.Series, factor_rets: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for f in factor_rets.columns:
        df = pd.concat([port, factor_rets[f]], axis=1).dropna()
        if len(df) < 3 or df.iloc[:, 1].var() == 0:
            rows[f] = {"Beta": np.nan, "R²": np.nan}
            continue
        beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
        rows[f] = {"Beta": float(beta), "R²": float(df.iloc[:, 0].corr(df.iloc[:, 1]) ** 2)}
    return pd.DataFrame(rows).T


def risk_on_off(factor_prices: pd.DataFrame, ma_window: int = 200, vix_cap: float = 20.0) -> pd.DataFrame:
    spy_ma = factor_prices["SPY"].rolling(ma_window).mean()
    qqq_ma = factor_prices["QQQ"].rolling(ma_window).mean()
    c_spy = factor_prices["SPY"] > spy_ma
    c_qqq = factor_prices["QQQ"] > qqq_ma
    c_vix = factor_prices["^VIX"] < vix_cap
    valid = spy_ma.notna() & qqq_ma.notna() & factor_prices["^VIX"].notna()
    on = (c_spy & c_qqq & c_vix).astype(float).where(valid)
    return pd.DataFrame({"SPY > MA": c_spy.astype(float).where(valid), "QQQ > MA": c_qqq.astype(float).where(valid),
                         "VIX < ngưỡng": c_vix.astype(float).where(valid), "risk_on": on})


def rolling_momentum_exposure(port: pd.Series, bench_prices: pd.Series, window: int = 20,
                              mom_window: int = 12) -> pd.Series:
    mom = bench_prices.pct_change(mom_window)
    df = pd.concat([port, mom], axis=1, join="inner").dropna()
    if len(df) < window:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


def highest_lowest_beta(beta_df: pd.DataFrame, k: int = 3) -> tuple[pd.Series, pd.Series]:
    b = beta_df.dropna(how="all")
    if b.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    last = b.iloc[-1].dropna()
    k = max(1, min(k, len(last)))
    return last.sort_values(ascending=False).head(k), last.sort_values().head(k)
