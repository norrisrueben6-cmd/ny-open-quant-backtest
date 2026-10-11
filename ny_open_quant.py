#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NY-OPEN QUANT DESK
=============================================================================
At the New York cash open (09:30 America/New_York, weekdays) this script runs
a full stack of quantitative finance models on QQQ (and converts the result to
NQ / MNQ futures terms), fuses them into ONE composite read for the session,
and posts it to a Discord channel through a webhook.

Models (each one returns a signal in [-1, +1], negative = bearish):

 1  Derivative pricing & stochastic calculus
      GBM + Ito's lemma | Black-Scholes-Merton + Greeks | Heston | SABR |
      Vasicek | CIR | Hull-White
 2  Statistical & time-series
      ARIMA / SARIMAX | (GJR-)GARCH | Engle-Granger & Johansen |
      Ornstein-Uhlenbeck | CAPM & Fama-French | Markowitz | Hidden Markov
 3  Numerical & computational
      Monte Carlo + variance reduction | Finite differences (explicit /
      implicit / Crank-Nicolson) | Newton & Brent root-finding |
      Binomial & trinomial lattices
 4  Machine learning & execution
      Hawkes processes | Avellaneda-Stoikov | Lasso / Ridge / ElasticNet |
      Random forest / gradient boosting (LightGBM, XGBoost if installed) |
      Q-learning (and optional PPO) execution agent

HONEST WARNING: none of this predicts the market.  Daily direction is close to
a coin flip; the composite is a weighted vote of statistical tilts, and each
model's weight is small.  The numbers that carry real information are the
expected-move / volatility estimates.  Treat the "bias" as context, not as a
trade signal.  Not financial advice.

-----------------------------------------------------------------------------
SETUP
    pip install numpy pandas scipy requests yfinance scikit-learn statsmodels arch hmmlearn
    # optional:  lightgbm xgboost pandas_market_calendars gymnasium stable-baselines3

    export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/XXXX/YYYY"

RUN
    python ny_open_quant.py                 # sleeps, fires at every NY open
    python ny_open_quant.py --now           # run once right now, send to Discord
    python ny_open_quant.py --now --dry-run # run once, print only
    python ny_open_quant.py --demo --now --dry-run   # synthetic data, no internet
    python ny_open_quant.py --now --ppo     # also train a PPO execution agent
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy import integrate, optimize, stats
from scipy.linalg import solve_banded
from scipy.special import ndtr as Ncdf

warnings.filterwarnings("ignore")
log = logging.getLogger("quant")


def _try(modname):
    try:
        return __import__(modname)
    except Exception:
        return None


yf = _try("yfinance")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    HAVE_SM = True
except Exception:
    HAVE_SM = False
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False
try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMM = True
except Exception:
    HAVE_HMM = False
try:
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False
lgb = _try("lightgbm")
xgb = _try("xgboost")

TD = 252.0                      # trading days per year
ET = ZoneInfo("America/New_York")


# =============================================================================
# CONFIG / CONTAINERS
# =============================================================================
@dataclass
class Config:
    webhook: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    equity: str = "QQQ"            # what the models run on
    futures: str = "NQ=F"          # used only to convert QQQ levels -> NQ/MNQ
    pair: str = "SPY"              # pairs-trading partner for QQQ
    years: int = 6                 # daily history to download
    lookback: int = 252            # default estimation window (trading days)
    run_delay_sec: int = 20        # seconds after 09:30:00 ET before firing
    div_yield: float = 0.006       # QQQ dividend yield used in BSM / lattices
    mc_paths: int = 200_000
    seed: int = 7
    as_gamma: float = 5e-4         # Avellaneda-Stoikov risk aversion
    as_k: float = 100.0            # Avellaneda-Stoikov order-arrival decay
    bias_threshold: float = 0.12   # |composite| above this -> UP / DOWN call
    as_of: Optional[str] = None    # 'YYYY-MM-DD' -> replay that day's open with only data known then
    # reversal-level trades in the morning card
    stop_pts: float = 35.0         # NQ points beyond the level
    target_pts: float = 150.0      # NQ points from the entry (TP2); TP1 = back to the open
    ppo: bool = False
    demo: bool = False
    dry_run: bool = False


@dataclass
class Result:
    key: str
    title: str
    section: str
    signal: float = 0.0
    weight: float = 0.0
    headline: str = ""
    lines: List[str] = field(default_factory=list)
    ok: bool = True


@dataclass
class Ctx:
    cfg: Config
    now: datetime
    px: pd.DataFrame                  # daily adjusted closes, all tickers, through PRIOR close
    ohlc: pd.DataFrame                # equity daily open/high/low/close/volume
    intra: pd.DataFrame               # equity 5-minute bars (recent sessions)
    spot: float                       # latest QQQ price (pre-market / open)
    fut_ratio: float                  # NQ / QQQ conversion ratio
    rf: float                         # risk-free (3m T-bill) decimal
    q: float                          # dividend yield
    near: Optional[dict] = None       # {'df','T','exp'}  nearest-expiry option chain
    term: Optional[dict] = None       # same, ~30 day expiry
    r: pd.Series = None               # daily log returns (close/close) of equity
    sig: Dict[str, float] = field(default_factory=dict)   # shared sigma estimates (daily, decimal)
    atm_iv: float = float("nan")
    atm_strike: float = float("nan")
    smile_near: Optional[pd.DataFrame] = None
    smile_term: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)
    info: Dict[str, object] = field(default_factory=dict)   # facts other models hand to the game plan
    premkt: Optional[pd.DataFrame] = None                    # today's pre-market 1-minute bars (live only)

    @property
    def prev_close(self) -> float:
        return float(self.ohlc["close"].iloc[-1])


# =============================================================================
# SMALL HELPERS
# =============================================================================
def clip(x, lo=-1.0, hi=1.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(max(lo, min(hi, x)))


def tsig(x, scale=1.0) -> float:
    return clip(math.tanh(x / scale))


def f(x, d=2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:,.{d}f}"


def pc(x, d=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.{d}f}%"


def bp(x, d=1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{1e4*x:.{d}f}bp"


def ols(y: np.ndarray, X: np.ndarray, const=True):
    """Plain OLS with classical standard errors -> beta, t-stats, R2, resid."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if const:
        X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    n, k = X.shape
    s2 = res @ res / max(n - k, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    r2 = 1 - (res @ res) / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return beta, beta / se, r2, res


# =============================================================================
# OPTION MATH  (Black-Scholes-Merton, Greeks, implied-vol root finders)
# =============================================================================
def d1d2(S, K, T, r, q, s):
    sT = s * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / sT
    return d1, d1 - sT


def bs_price(S, K, T, r, q, s, kind="c"):
    if T <= 0 or s <= 0:
        return max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
    d1, d2 = d1d2(S, K, T, r, q, s)
    if kind == "c":
        return S * math.exp(-q * T) * Ncdf(d1) - K * math.exp(-r * T) * Ncdf(d2)
    return K * math.exp(-r * T) * Ncdf(-d2) - S * math.exp(-q * T) * Ncdf(-d1)


def bs_vega_raw(S, K, T, r, q, s):
    d1, _ = d1d2(S, K, T, r, q, s)
    return S * math.exp(-q * T) * stats.norm.pdf(d1) * math.sqrt(T)


def bs_greeks(S, K, T, r, q, s, kind="c") -> dict:
    d1, d2 = d1d2(S, K, T, r, q, s)
    pdf = stats.norm.pdf(d1)
    eq, er, sT = math.exp(-q * T), math.exp(-r * T), math.sqrt(T)
    if kind == "c":
        delta = eq * Ncdf(d1)
        theta = (-S * eq * pdf * s / (2 * sT) - r * K * er * Ncdf(d2) + q * S * eq * Ncdf(d1))
        rho = K * T * er * Ncdf(d2)
    else:
        delta = -eq * Ncdf(-d1)
        theta = (-S * eq * pdf * s / (2 * sT) + r * K * er * Ncdf(-d2) - q * S * eq * Ncdf(-d1))
        rho = -K * T * er * Ncdf(-d2)
    gamma = eq * pdf / (S * s * sT)
    vega = S * eq * pdf * sT
    vanna = -eq * pdf * d2 / s
    volga = vega * d1 * d2 / s
    charm = -eq * (pdf * (2 * (r - q) * T - d2 * s * sT) / (2 * T * s * sT) + (q * Ncdf(d1) if kind == "c" else -q * Ncdf(-d1)))
    return dict(delta=delta, gamma=gamma, theta_day=theta / 365.0, vega_pt=vega / 100.0,
                rho_pt=rho / 100.0, vanna=vanna, volga=volga, charm_day=charm / 365.0)


def _intrinsic_ok(price, S, K, T, r, q, kind):
    lo = max((S * math.exp(-q * T) - K * math.exp(-r * T)) if kind == "c" else (K * math.exp(-r * T) - S * math.exp(-q * T)), 0.0)
    return price > lo + 1e-9


def iv_newton(price, S, K, T, r, q, kind, x0=0.25, tol=1e-9, maxit=60):
    """Newton-Raphson on  f(s) = BS(s) - price,  f'(s) = vega.  Returns (iv, iterations)."""
    if not _intrinsic_ok(price, S, K, T, r, q, kind):
        return float("nan"), 0
    s = x0
    for i in range(1, maxit + 1):
        diff = bs_price(S, K, T, r, q, s, kind) - price
        if abs(diff) < tol:
            return s, i
        v = bs_vega_raw(S, K, T, r, q, s)
        if v < 1e-10:
            break
        s = min(max(s - diff / v, 1e-4), 8.0)
    return float("nan"), maxit


def iv_brent(price, S, K, T, r, q, kind):
    """Brent's method (bracketing, super-linear, never diverges).  Returns (iv, iterations)."""
    if not _intrinsic_ok(price, S, K, T, r, q, kind):
        return float("nan"), 0
    g = lambda s: bs_price(S, K, T, r, q, s, kind) - price
    try:
        s, info = optimize.brentq(g, 1e-4, 8.0, xtol=1e-10, full_output=True)
        return s, info.iterations
    except Exception:
        return float("nan"), 0


def iv_bisect(price, S, K, T, r, q, kind, tol=1e-9):
    if not _intrinsic_ok(price, S, K, T, r, q, kind):
        return float("nan"), 0
    lo, hi, n = 1e-4, 8.0, 0
    while hi - lo > tol and n < 200:
        mid = 0.5 * (lo + hi)
        n += 1
        if bs_price(S, K, T, r, q, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), n


# =============================================================================
# MARKET DATA
# =============================================================================
def _flat(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _mid(o) -> float:
    bid, ask, last = float(o.get("bid", 0) or 0), float(o.get("ask", 0) or 0), float(o.get("lastPrice", 0) or 0)
    return 0.5 * (bid + ask) if bid > 0 and ask > 0 else last


def _build_smile(chain: dict, S, r, q) -> Tuple[pd.DataFrame, float, float]:
    """OTM implied-vol smile (Brent): puts below spot, calls above spot. Plus ATM strike / ATM IV."""
    df, T = chain["df"], chain["T"]
    rows = []
    for _, o in df.iterrows():
        K, kind = float(o["strike"]), o["kind"]
        if (kind == "c" and K < S) or (kind == "p" and K > S):
            continue                                  # skip in-the-money side (wide spreads, stale prints)
        mid = _mid(o)
        if mid <= 0.02:
            continue
        iv, _ = iv_brent(mid, S, K, T, r, q, kind)
        if not math.isnan(iv):
            rows.append(dict(K=K, iv=iv, kind=kind, mid=mid, vol=float(o.get("volume", 0) or 0), oi=float(o.get("openInterest", 0) or 0)))
    sm = pd.DataFrame(rows)
    if sm.empty:
        return sm, float("nan"), float("nan")
    k0 = sm.iloc[(sm["K"] - S).abs().argsort()[:1]]["K"].iloc[0]
    atm = sm[(sm["K"] - k0).abs() < 1e-9]["iv"]
    # ATM: average call and put IV at the strike nearest spot
    both = []
    for kind in ("c", "p"):
        sub = df[(df["strike"] == k0) & (df["kind"] == kind)]
        if len(sub):
            mid = _mid(sub.iloc[0])
            if mid > 0.02:
                iv, _ = iv_brent(mid, S, k0, T, r, q, kind)
                if not math.isnan(iv):
                    both.append(iv)
    atm_iv = float(np.mean(both)) if both else float(atm.mean())
    return sm.sort_values("K").reset_index(drop=True), float(k0), atm_iv


def _load_chain(tk, today, now, which="near"):
    exps = list(tk.options)
    if not exps:
        return None
    dts = [(datetime.strptime(e, "%Y-%m-%d").date(), e) for e in exps]
    if which == "near":
        cand = [e for d, e in dts if d >= today]
    else:  # ~ 30 calendar days out
        cand = sorted([e for d, e in dts if (d - today).days >= 21], key=lambda e: e)
    if not cand:
        return None
    exp = cand[0]
    oc = tk.option_chain(exp)
    df = pd.concat([oc.calls.assign(kind="c"), oc.puts.assign(kind="p")], ignore_index=True)
    close_dt = datetime.strptime(exp, "%Y-%m-%d").replace(hour=16, tzinfo=ET)
    T = max((close_dt - now).total_seconds() / (365.0 * 86400.0), 1.0 / (365 * 24))
    return dict(df=df, T=T, exp=exp)


def load_live(cfg: Config) -> Ctx:
    if yf is None:
        raise RuntimeError("yfinance is not installed (pip install yfinance), or use --demo")
    now = datetime.now(ET)
    if cfg.as_of:
        d0 = datetime.strptime(cfg.as_of, "%Y-%m-%d").date()
        now = datetime(d0.year, d0.month, d0.day, 9, 30, 20, tzinfo=ET)
    today = now.date()
    tickers = [cfg.equity, cfg.pair, "IWM", "TLT", "GLD", "IWD", "IWF", "MTUM", "^VIX", "^IRX", "^FVX", "^TNX", "^TYX", cfg.futures] + EXTRA_TICKERS
    tickers = list(dict.fromkeys(tickers))
    if cfg.as_of:
        start = (today - timedelta(days=int(365.25 * cfg.years) + 10)).isoformat()
        end = (today + timedelta(days=5)).isoformat()
        raw = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=True, progress=False, group_by="column")
    else:
        raw = yf.download(tickers, period=f"{cfg.years}y", interval="1d", auto_adjust=True, progress=False, group_by="column")
    close = raw["Close"].copy()
    close = close[close.index.date < today].ffill()
    ohlc = pd.DataFrame({k.lower(): raw[k][cfg.equity] for k in ["Open", "High", "Low", "Close", "Volume"]})
    ohlc = ohlc[ohlc.index.date < today].dropna()
    px = close.reindex(ohlc.index).ffill()
    # latest price (pre-market / open)
    spot = float(ohlc["close"].iloc[-1])
    actual = None
    premkt_bars = None
    if cfg.as_of:
        day = raw.loc[raw.index.date == today]
        if day.empty or pd.isna(day["Open"][cfg.equity].iloc[0]):
            raise RuntimeError(f"{today} was not a trading day for {cfg.equity} (or Yahoo has no data for it)")
        spot = float(day["Open"][cfg.equity].iloc[0])
        actual = {k: float(day[k.capitalize()][cfg.equity].iloc[0]) for k in ("open", "high", "low", "close")}
    try:
        if cfg.as_of:
            raise RuntimeError("replay mode: using that day's official open")
        m1 = _flat(yf.download(cfg.equity, period="1d", interval="1m", prepost=True, auto_adjust=True, progress=False))
        if len(m1):
            spot = float(m1["Close"].dropna().iloc[-1])
            idx = m1.index.tz_convert(ET) if m1.index.tz is not None else m1.index.tz_localize(ET)
            pm = m1[(idx.date == today) & (idx.time < datetime(2000, 1, 1, 9, 30).time())]
            premkt_bars = pm.rename(columns=str.lower) if len(pm) else None
    except Exception as e:
        if not cfg.as_of:
            log.warning("1m spot fetch failed: %s", e)
    intra = pd.DataFrame()
    try:
        intra = _flat(yf.download(cfg.equity, period="60d" if cfg.as_of else "30d", interval="5m", auto_adjust=True, progress=False))
        intra = intra.rename(columns=str.lower)
        intra = intra[intra.index.tz_convert(ET).date < today] if intra.index.tz is not None else intra
        if cfg.as_of:
            intra = intra[intra.index.tz_convert(ET).date >= today - timedelta(days=45)] if len(intra) else intra
    except Exception as e:
        log.warning("5m fetch failed: %s", e)
    fut_ratio = float("nan")
    try:
        fut = float(px[cfg.futures].dropna().iloc[-1])
        fut_ratio = fut / float(px[cfg.equity].iloc[-1])
    except Exception:
        pass
    rf = float(px["^IRX"].dropna().iloc[-1]) / 100.0 if "^IRX" in px else 0.04
    ctx = Ctx(cfg=cfg, now=now, px=px, ohlc=ohlc, intra=intra, spot=spot, fut_ratio=fut_ratio, rf=rf, q=cfg.div_yield)
    ctx.premkt = premkt_bars
    if cfg.as_of:
        ctx.info["actual"] = actual
        ctx.notes.append(f"REPLAY of {today}: only data available before that open was used. "
                         "Historical option chains aren't free, so implied vol comes from VIX.")
        if len(intra) < 600:
            ctx.notes.append("Intraday 5-minute history only goes back ~60 days on Yahoo, so the Hawkes model may be skipped for older dates.")
        return ctx
    try:
        tk = yf.Ticker(cfg.equity)
        ctx.near = _load_chain(tk, today, now, "near")
        ctx.term = _load_chain(tk, today, now, "term")
    except Exception as e:
        log.warning("option chain fetch failed: %s", e)
    return ctx


def finish_ctx(c: Ctx) -> Ctx:
    c.r = np.log(c.ohlc["close"]).diff().dropna()
    c.atm_strike = float(round(c.spot))
    for name in ("near", "term"):
        ch = getattr(c, name)
        if ch is not None:
            sm, k0, iv = _build_smile(ch, c.spot, c.rf, c.q)
            setattr(c, "smile_" + name, sm)
            if name == "near" and not math.isnan(iv):
                c.atm_iv, c.atm_strike = iv, k0
    if math.isnan(c.atm_iv):
        # fallback: scale VIX (SPX vol) to Nasdaq vol
        vix = float(c.px["^VIX"].dropna().iloc[-1]) / 100.0 if "^VIX" in c.px else 0.18
        c.atm_iv = vix * 1.2
        c.notes.append("No option chain: ATM IV approximated as 1.2 x VIX.")
    return c


# =============================================================================
# SYNTHETIC DATA  (--demo : lets you test the whole pipeline with no internet)
# =============================================================================
def _sabr_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied-volatility approximation."""
    if abs(F - K) < 1e-9 * F:
        A = ((1 - beta) ** 2 / 24) * alpha ** 2 / F ** (2 - 2 * beta) + 0.25 * rho * beta * nu * alpha / F ** (1 - beta) + (2 - 3 * rho ** 2) / 24 * nu ** 2
        return alpha / F ** (1 - beta) * (1 + A * T)
    lfk = math.log(F / K)
    fk = (F * K) ** ((1 - beta) / 2)
    z = nu / alpha * fk * lfk
    xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    zx = 1.0 if abs(z) < 1e-8 else z / xz
    A = 1 + ((1 - beta) ** 2 / 24) * lfk ** 2 + ((1 - beta) ** 4 / 1920) * lfk ** 4
    B = 1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / fk ** 2 + 0.25 * rho * beta * nu * alpha / fk + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    return alpha / (fk * A) * zx * B


def make_demo(cfg: Config) -> Ctx:
    rng = np.random.default_rng(cfg.seed)
    now = datetime.now(ET)
    n = cfg.years * 252
    idx = pd.bdate_range(end=(now - timedelta(days=1)).date(), periods=n)
    n = len(idx)                                   # pandas can return one fewer day near weekends
    dt = 1 / TD
    kappa, theta, xi, rho = 4.0, 0.045, 0.7, -0.75
    v, ret, vs = 0.04, np.empty(n), np.empty(n)
    for t in range(n):
        z1 = rng.standard_normal()
        z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.standard_normal()
        ret[t] = (0.11 - 0.5 * v) * dt + math.sqrt(max(v, 1e-6) * dt) * z1
        v = max(v + kappa * (theta - v) * dt + xi * math.sqrt(max(v, 1e-6) * dt) * z2, 2e-3)
        vs[t] = v
    q = 480 * np.exp(np.cumsum(ret) - ret.sum())
    mk = ret
    px = pd.DataFrame(index=idx)
    px["QQQ"] = q
    px["SPY"] = 520 * np.exp(np.cumsum(0.85 * mk + 0.004 * rng.standard_normal(n)))
    px["IWM"] = 210 * np.exp(np.cumsum(0.9 * mk + 0.008 * rng.standard_normal(n)))
    px["TLT"] = 95 * np.exp(np.cumsum(-0.15 * mk + 0.006 * rng.standard_normal(n)))
    px["GLD"] = 230 * np.exp(np.cumsum(0.03 * mk + 0.007 * rng.standard_normal(n)))
    px["IWD"] = 190 * np.exp(np.cumsum(0.75 * mk + 0.004 * rng.standard_normal(n)))
    px["IWF"] = 400 * np.exp(np.cumsum(0.95 * mk + 0.003 * rng.standard_normal(n)))
    px["MTUM"] = 220 * np.exp(np.cumsum(0.9 * mk + 0.005 * rng.standard_normal(n)))
    px["^VIX"] = 100 * np.sqrt(vs) * 0.78 + rng.normal(0, 0.6, n)
    short = np.empty(n); short[0] = 4.3
    for t in range(1, n):
        short[t] = max(short[t - 1] + 1.5 * (4.3 - short[t - 1]) * dt + 0.25 * math.sqrt(dt) * rng.standard_normal(), 0.05)
    px["^IRX"] = short
    px["^FVX"] = 4.0 + 0.4 * (short - 4.3) + rng.normal(0, 0.03, n).cumsum() * 0.05
    px["^TNX"] = px["^FVX"] + 0.25
    px["^TYX"] = px["^FVX"] + 0.55
    px["NQ=F"] = px["QQQ"] * 41.6
    gap = rng.normal(0, 0.003, n)
    op = px["QQQ"].shift(1).fillna(px["QQQ"].iloc[0]) * np.exp(gap)
    cl = px["QQQ"]
    hi = np.maximum(op, cl) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(op, cl) * (1 - np.abs(rng.normal(0, 0.003, n)))
    ohlc = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": cl, "volume": rng.lognormal(17.5, 0.25, n)}, index=idx)
    # intraday 5-minute bars, self-exciting
    bars, tidx = [], []
    last_px, inten = float(cl.iloc[-1]), 0.0
    sess = idx[-30:]
    for d in sess:
        t0 = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
        for b in range(78):
            u = 1 + 0.9 * abs(b - 39) / 39
            sg = math.sqrt(vs[-1] * dt / 78) * u * (1 + inten)
            shock = rng.standard_t(4) / math.sqrt(2)
            rr = sg * shock
            inten = inten * math.exp(-0.15) + (0.8 if abs(shock) > 2.4 else 0.0)
            o_, c_ = last_px, last_px * math.exp(rr)
            bars.append((o_, max(o_, c_) * 1.0004, min(o_, c_) * 0.9996, c_, rng.lognormal(12.5, 0.4) * (1 + inten)))
            tidx.append(t0 + timedelta(minutes=5 * b))
            last_px = c_
    intra = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(tidx))
    spot = float(cl.iloc[-1] * math.exp(rng.normal(0, 0.002)))
    rf = float(px["^IRX"].iloc[-1]) / 100
    c = Ctx(cfg=cfg, now=now, px=px, ohlc=ohlc, intra=intra, spot=spot, fut_ratio=41.6, rf=rf, q=cfg.div_yield)

    def synth_chain(T, exp, beta=0.5, rho_s=-0.35, nu=1.1):
        F = spot * math.exp((rf - c.q) * T)
        atm = 0.19
        alpha = atm * F ** (1 - beta)
        rows = []
        for K in np.arange(round(spot * 0.85), round(spot * 1.15) + 1, 1.0):
            vol = _sabr_vol(F, K, T, alpha, beta, rho_s, nu)
            for kind in ("c", "p"):
                m = bs_price(spot, K, T, rf, c.q, vol, kind)
                if m < 0.03:
                    continue
                rows.append(dict(strike=K, kind=kind, bid=max(m * 0.995 - 0.005, 0.0), ask=m * 1.005 + 0.005, lastPrice=m, volume=100, openInterest=500))
        return dict(df=pd.DataFrame(rows), T=T, exp=exp)

    c.near = synth_chain(6.5 / 24 / 365, now.strftime("%Y-%m-%d"))
    c.term = synth_chain(30 / 365, (now + timedelta(days=30)).strftime("%Y-%m-%d"))
    c.notes.append("DEMO MODE: all data is synthetic.")
    return c


# =============================================================================
# SECTION 1 -- DERIVATIVE PRICING & STOCHASTIC CALCULUS
# =============================================================================
def mod_gbm_ito(c: Ctx) -> Result:
    R = Result("gbm", "Geometric Brownian Motion & Ito's Lemma", "1. Derivative pricing & stochastic calculus", weight=0.3)
    r = c.r.iloc[-c.cfg.lookback:]
    dt = 1 / TD
    m_d, s_d = float(r.mean()), float(r.std(ddof=1))
    sig = s_d * math.sqrt(TD)
    mu = m_d * TD + 0.5 * sig ** 2
    drift, vol = (mu - 0.5 * sig ** 2) * dt, sig * math.sqrt(dt)
    p_dn = float(Ncdf(-drift / vol))
    z05 = stats.norm.ppf(0.05)
    q05, q95 = c.spot * math.exp(drift + vol * z05), c.spot * math.exp(drift - vol * z05)
    ito_corr = -0.5 * sig ** 2 * dt
    rng = np.random.default_rng(c.cfg.seed)
    ST = c.spot * np.exp(drift + vol * rng.standard_normal(400_000))
    mc_log, mc_lin = float(np.mean(np.log(ST / c.spot))), float(np.mean(ST / c.spot - 1))
    ito_sq = (2 * mu + sig ** 2) * dt
    mc_sq = float(np.mean((ST / c.spot) ** 2 - 1))
    c.sig["gbm"] = s_d
    R.signal = clip((1 - p_dn) * 2 - 1)
    R.headline = f"mu={pc(mu)}/yr  sigma={pc(sig)}/yr  1d sigma={pc(s_d)}  P(down)={pc(p_dn,1)}  90% range {f(q05)}-{f(q95)}"
    R.lines = [
        "SDE:      dS = mu*S*dt + sigma*S*dW   (W = Wiener process)",
        f"Estimates ({len(r)}d):  mean daily log-ret={bp(m_d,2)}  daily sd={pc(s_d,3)}  sigma={pc(sig)}  mu={pc(mu)}",
        "Ito's lemma on f=ln S:   d ln S = (mu - sigma^2/2) dt + sigma dW",
        f"   Ito convexity term -sigma^2/2*dt = {bp(ito_corr,3)}/day  (log-drift = {bp(drift,3)}/day)",
        f"   MC check (400k draws):  E[ln(S1/S0)] = {bp(mc_log,3)}   E[S1/S0 - 1] = {bp(mc_lin,3)}  vs  mu*dt = {bp(mu*dt,3)}",
        "Ito's lemma on f=S^2:    d(S^2) = (2 mu + sigma^2) S^2 dt + 2 sigma S^2 dW",
        f"   analytic E[(S1/S0)^2 - 1] = {bp(ito_sq,3)}   MC = {bp(mc_sq,3)}",
        "Black-Scholes PDE (from Ito + delta-hedge):  dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r-q) S dV/dS - rV = 0",
        f"1-day law:  ln(S1/S0) ~ N({bp(drift,3)}, {pc(vol,3)}^2)    P(S1<S0) = {pc(p_dn,2)}",
        f"   5% / 95% quantiles of S1: {f(q05)} / {f(q95)}",
    ]
    return R


def mod_bsm(c: Ctx) -> Result:
    R = Result("bsm", "Black-Scholes-Merton & the Greeks", "1. Derivative pricing & stochastic calculus", weight=0.0)
    S, r, q = c.spot, c.rf, c.q
    T = c.near["T"] if c.near else 6.5 / 24 / 365
    K, sig = c.atm_strike, c.atm_iv
    call, put = bs_price(S, K, T, r, q, sig, "c"), bs_price(S, K, T, r, q, sig, "p")
    gc, gp = bs_greeks(S, K, T, r, q, sig, "c"), bs_greeks(S, K, T, r, q, sig, "p")
    straddle = call + put
    exp_abs = straddle
    sd_move = straddle * math.sqrt(math.pi / 2)           # E|X| = sigma*sqrt(2/pi)  ->  sigma = E|X| * sqrt(pi/2)
    c.sig["implied_move"] = sd_move / S
    parity = (call - put) - (S * math.exp(-q * T) - K * math.exp(-r * T))
    rv21 = float(c.r.iloc[-21:].std(ddof=1)) * math.sqrt(TD)
    vrp = sig - rv21
    c.info.update(iv=sig, rv21=rv21, vrp=vrp)
    gsum = gc["gamma"] + gp["gamma"]
    gamma_be = math.sqrt(2 * abs(gc["theta_day"] + gp["theta_day"]) / max(gsum, 1e-12)) if gsum > 0 else float("nan")
    d1, d2 = d1d2(S, K, T, r, q, sig)
    R.headline = (f"ATM K={f(K,0)} IV={pc(sig,1)}  straddle=${f(straddle)} -> implied 1-sigma move +/-${f(sd_move)} ({pc(sd_move/S)})  "
                  f"IV-RV21={pc(vrp,1)}")
    R.lines = [
        "C = S e^{-qT} N(d1) - K e^{-rT} N(d2);  P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)",
        "d1 = [ln(S/K) + (r - q + sigma^2/2)T] / (sigma sqrt T);   d2 = d1 - sigma sqrt T",
        f"Inputs: S={f(S)} K={f(K,0)} T={T*365*24:.1f}h ({T:.5f}y) r={pc(r)} q={pc(q)} sigma(IV)={pc(sig,2)}   d1={d1:.4f} d2={d2:.4f}",
        f"Call={f(call,3)}  Put={f(put,3)}  Straddle={f(straddle,3)}   put-call parity residual={parity:.2e}",
        f"Implied 1-sigma move = straddle*sqrt(pi/2) = ${f(sd_move)}  ({pc(sd_move/S)})   expected |move| ~ ${f(exp_abs)}",
        "GREEKS (call | put):",
        f"   Delta {gc['delta']:+.4f} | {gp['delta']:+.4f}      Gamma {gc['gamma']:.5f} | {gp['gamma']:.5f}",
        f"   Theta/day {gc['theta_day']:+.4f} | {gp['theta_day']:+.4f}      Vega/1pt {gc['vega_pt']:.4f} | {gp['vega_pt']:.4f}      Rho/1pt {gc['rho_pt']:+.4f} | {gp['rho_pt']:+.4f}",
        f"   Vanna {gc['vanna']:+.4f}   Volga {gc['volga']:+.4f}   Charm/day {gc['charm_day']:+.5f}  (2nd-order Greeks)",
        f"Delta-neutral hedge for 100 long calls: short {100*100*gc['delta']:.0f} shares; residual gamma P&L for a $1 move = {0.5*100*100*gc['gamma']:.2f} $",
        f"Gamma/theta break-even daily move of a long straddle = ${f(gamma_be)}  (= {pc(gamma_be/S)})  vs implied sd ${f(sd_move)}",
        f"Variance risk premium: IV {pc(sig,1)} - 21d realised {pc(rv21,1)} = {pc(vrp,1)}   ({'options rich' if vrp>0 else 'options cheap'})",
    ]
    return R


def heston_cf(u, T, r, q, S0, v0, kappa, theta, xi, rho):
    """Heston characteristic function of ln S_T (Albrecher et al. 'little trap' form)."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi ** 2 * (iu + u * u))
    g = (b - d) / (b + d)
    edT = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * edT) / (1 - g)))
    D = (b - d) / xi ** 2 * (1 - edT) / (1 - g * edT)
    return np.exp(C + D * v0 + iu * math.log(S0))


def heston_call(S, K, T, r, q, v0, kappa, theta, xi, rho):
    u = np.linspace(1e-6, 150.0, 3001)
    lnK = math.log(K)
    phi = lambda uu: heston_cf(uu, T, r, q, S, v0, kappa, theta, xi, rho)
    ES = float(np.real(phi(np.array([-1j]))[0]))                      # = S e^{(r-q)T}
    I1 = np.real(np.exp(-1j * u * lnK) * phi(u - 1j) / (1j * u * ES))
    I2 = np.real(np.exp(-1j * u * lnK) * phi(u) / (1j * u))
    P1 = 0.5 + integrate.simpson(I1, x=u) / math.pi
    P2 = 0.5 + integrate.simpson(I2, x=u) / math.pi
    return S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2


def heston_mc(S0, v0, kappa, theta, xi, rho, mu, T, n=100_000, steps=78, seed=1):
    """Full-truncation Euler with antithetic variates."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    h = n // 2
    lnS, v = np.full(2 * h, math.log(S0)), np.full(2 * h, v0)
    for _ in range(steps):
        z1 = rng.standard_normal(h)
        z2 = rng.standard_normal(h)
        z1, z2 = np.concatenate([z1, -z1]), np.concatenate([z2, -z2])
        w2 = rho * z1 + math.sqrt(1 - rho ** 2) * z2
        vp = np.maximum(v, 0.0)
        lnS += (mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * w2
    return np.exp(lnS)


def mod_heston(c: Ctx) -> Result:
    R = Result("heston", "Heston stochastic-volatility model", "1. Derivative pricing & stochastic calculus", weight=0.4)
    r = c.r.iloc[-750:]
    rv = (r ** 2).rolling(10).mean().dropna() * TD            # annualised variance proxy
    v, dt = rv.values, 1 / TD
    dv, vl = np.diff(v), v[:-1]
    (a0, b0), _, _, res = ols(dv, vl)
    kappa = float(np.clip(-b0 / dt, 0.5, 15.0))
    theta = float(np.clip(-a0 / b0 if b0 < 0 else v.mean(), 0.015, 0.4))
    xi = float(np.clip(np.std(res / np.sqrt(np.maximum(vl, 1e-6))) / math.sqrt(dt), 0.2, 1.8))
    dvix = np.log(c.px["^VIX"]).diff().reindex(r.index)
    both = pd.concat([r, dvix], axis=1).dropna()
    rho = float(np.clip(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1], -0.95, 0.3))
    lam = 0.94
    ew = float(np.sqrt(np.average(r.iloc[-250:].values ** 2, weights=lam ** np.arange(250)[::-1])) * math.sqrt(TD)) ** 2
    v0 = ew
    feller = 2 * kappa * theta - xi ** 2
    mu = 0.08                                                  # equity-premium prior for the physical drift
    ST = heston_mc(c.spot, v0, kappa, theta, xi, rho, mu, 1 / TD, n=c.cfg.mc_paths // 2, seed=c.cfg.seed)
    ret = np.log(ST / c.spot)
    p_up = float((ret > 0).mean())
    q05, es = float(np.quantile(ret, 0.05)), float(ret[ret <= np.quantile(ret, 0.05)].mean())
    sd1 = float(ret.std())
    c.sig["heston"] = sd1
    lines = [
        "dS = (r-q) S dt + sqrt(v) S dW1;   dv = kappa (theta - v) dt + xi sqrt(v) dW2;   dW1 dW2 = rho dt",
        "Historical calibration (10d realised variance, OLS on dv = a + b v dt + xi sqrt(v) eps; rho = corr(return, dlnVIX)):",
        f"   kappa={kappa:.2f} (half-life of variance {math.log(2)/kappa*TD:.0f}d)  theta={theta:.4f} (long-run vol {pc(math.sqrt(theta),1)})  xi={xi:.2f}  rho={rho:.2f}",
        f"   v0 (EWMA)={v0:.4f} (vol {pc(math.sqrt(v0),1)})   Feller 2k*theta - xi^2 = {feller:.3f} ({'variance stays > 0' if feller>0 else 'variance can touch 0'})",
        f"1-day full-truncation Euler MC ({len(ST):,} antithetic paths, 78 steps): sd={pc(sd1,3)}  P(up)={pc(p_up,1)}  5% quantile={pc(q05,2)}  ES5={pc(es,2)}",
    ]
    try:
        T = 30 / 365
        K = c.spot
        hp = heston_call(c.spot, K, T, c.rf, c.q, v0, kappa, theta, xi, rho)
        bsp = bs_price(c.spot, K, T, c.rf, c.q, math.sqrt(v0), "c")
        hiv = lambda k: iv_brent(heston_call(c.spot, k, T, c.rf, c.q, v0, kappa, theta, xi, rho), c.spot, k, T, c.rf, c.q, "c")[0]
        dn, up = hiv(0.95 * c.spot), hiv(1.05 * c.spot)
        skew = dn - up
        lines.append(f"30d ATM call: Heston(CF integral)={f(hp,3)}  vs BS(sigma=sqrt v0)={f(bsp,3)}")
        lines.append(f"Heston smile: IV(95%K)={pc(dn,2)}  IV(105%K)={pc(up,2)}  skew={pc(skew,2)}  (rho<0 => put skew)")
        if c.smile_term is not None and len(c.smile_term) > 6:
            sm = c.smile_term
            mi = lambda k: float(np.interp(k, sm["K"], sm["iv"]))
            lines.append(f"Market 30d smile: IV(95%K)={pc(mi(0.95*c.spot),2)}  IV(105%K)={pc(mi(1.05*c.spot),2)}  skew={pc(mi(0.95*c.spot)-mi(1.05*c.spot),2)}")
    except Exception as e:
        lines.append(f"(semi-analytic pricing skipped: {e})")
    c.info.update(heston_rho=rho, heston_xi=xi, heston_v0vol=math.sqrt(v0), heston_thetavol=math.sqrt(theta))
    R.signal = clip((p_up - 0.5) * 2)
    R.headline = f"kappa={kappa:.1f} theta-vol={pc(math.sqrt(theta),0)} xi={xi:.2f} rho={rho:.2f} v0-vol={pc(math.sqrt(v0),1)} | 1d sd={pc(sd1,2)} P(up)={pc(p_up,1)} ES5={pc(es,2)}"
    R.lines = lines
    return R


def sabr_calibrate(F, T, Ks, ivs, beta=0.5):
    Ks, ivs = np.asarray(Ks, float), np.asarray(ivs, float)
    atm = float(np.interp(F, Ks, ivs))
    w = np.exp(-((np.log(Ks / F) / 0.06) ** 2)) + 0.05
    a0 = atm * F ** (1 - beta)

    def resid(p):
        a, rh, nu = p
        return [w[i] * (_sabr_vol(F, Ks[i], T, a, beta, rh, nu) - ivs[i]) for i in range(len(Ks))]
    sol = optimize.least_squares(resid, [a0, -0.3, 0.8], bounds=([1e-4, -0.999, 1e-3], [20 * a0 + 1, 0.999, 6.0]))
    a, rh, nu = sol.x
    fit = np.array([_sabr_vol(F, Ks[i], T, a, beta, rh, nu) for i in range(len(Ks))])
    rmse = float(np.sqrt(np.mean((fit - ivs) ** 2)))
    return a, rh, nu, rmse


def mod_sabr(c: Ctx) -> Result:
    R = Result("sabr", "SABR smile model", "1. Derivative pricing & stochastic calculus", weight=0.4)
    out = []
    for name in ("term", "near"):
        sm = getattr(c, "smile_" + name)
        ch = getattr(c, name)
        if sm is None or ch is None or len(sm) < 7:
            continue
        T = ch["T"]
        F = c.spot * math.exp((c.rf - c.q) * T)
        atm = float(np.interp(F, sm["K"], sm["iv"]))
        width = max(0.04, 2.5 * atm * math.sqrt(T))
        s2 = sm[np.abs(np.log(sm["K"] / F)) < width].groupby("K", as_index=False)["iv"].mean()
        if len(s2) < 6:
            continue
        a, rh, nu, rmse = sabr_calibrate(F, T, s2["K"].values, s2["iv"].values, 0.5)
        d = max(0.01, atm * math.sqrt(T))
        s_dn, s_up = _sabr_vol(F, F * math.exp(-d), T, a, 0.5, rh, nu), _sabr_vol(F, F * math.exp(d), T, a, 0.5, rh, nu)
        rr, bf = s_dn - s_up, 0.5 * (s_dn + s_up) - atm
        out.append((name, ch["exp"], T, F, atm, a, rh, nu, rmse, rr, bf, d, len(s2)))
    if not out:
        raise RuntimeError("no usable option smile (need >=6 OTM strikes)")
    R.lines = ["sigma_B(K,F,T) = alpha/[(FK)^((1-b)/2) (1+(1-b)^2/24 ln^2(F/K) + ..)] * z/x(z) * [1 + ((1-b)^2 alpha^2/(24 (FK)^(1-b)) + rho b nu alpha/(4 (FK)^((1-b)/2)) + (2-3 rho^2) nu^2/24) T]",
               "z = nu/alpha (FK)^((1-b)/2) ln(F/K);   x(z) = ln[(sqrt(1-2 rho z+z^2)+z-rho)/(1-rho)];   beta fixed = 0.5; calibrate (alpha, rho, nu) by weighted least squares"]
    for (name, exp, T, F, atm, a, rh, nu, rmse, rr, bf, d, npts) in out:
        R.lines.append(f"[{name} {exp}  T={T*365:.2f}d  F={f(F)}  {npts} strikes]  alpha={a:.4f} rho={rh:.3f} nu={nu:.3f}  fit RMSE={100*rmse:.2f} vol-pts")
        R.lines.append(f"     ATM vol={pc(atm,2)}  1-sigma-strike ({pc(d,2)}) risk-reversal(put-call)={pc(rr,2)}  butterfly={pc(bf,2)}  skew/ATM={rr/atm:.3f}")
    nm, exp, T, F, atm, a, rh, nu, rmse, rr, bf, d, npts = out[0]
    ratio = rr / atm
    c.info.update(sabr_ratio=ratio, sabr_rho=rh, sabr_nu=nu)
    R.signal = clip(-0.4 * math.tanh((ratio - 0.10) / 0.08))      # richer-than-usual put skew -> bearish tilt
    R.headline = f"[{nm}] rho={rh:.2f} nu={nu:.2f} alpha={a:.3f} | put-skew/ATM={ratio:.2f} (neutral ~0.10) | fit {100*rmse:.2f}vp"
    return R


# ---------- short-rate models ----------
def vasicek_yield(r0, a, b, s, T):
    B = (1 - math.exp(-a * T)) / a
    A = math.exp((b - s * s / (2 * a * a)) * (B - T) - s * s * B * B / (4 * a))
    return -math.log(A * math.exp(-B * r0)) / T


def cir_yield(r0, a, b, s, T):
    g = math.sqrt(a * a + 2 * s * s)
    e = math.exp(g * T) - 1
    den = (g + a) * e + 2 * g
    B = 2 * e / den
    A = (2 * g * math.exp((a + g) * T / 2) / den) ** (2 * a * b / (s * s))
    return -math.log(A * math.exp(-B * r0)) / T


def mod_rates(c: Ctx) -> Result:
    R = Result("rates", "Vasicek / CIR / Hull-White short-rate models", "1. Derivative pricing & stochastic calculus", weight=0.3)
    rs = (c.px["^IRX"].dropna().iloc[-750:] / 100.0).values
    dt = 1 / TD
    # Vasicek: exact AR(1) discretisation
    (al, ph), _, _, res = ols(rs[1:], rs[:-1])
    ph = min(max(ph, 0.5), 0.99999)
    a_v = -math.log(ph) / dt
    b_v = al / (1 - ph)
    s_v = float(np.std(res, ddof=2)) * math.sqrt(2 * a_v / (1 - ph ** 2))
    # CIR: dr/sqrt(r) regression
    rpos = np.maximum(rs[:-1], 1e-4)
    y = np.diff(rs) / np.sqrt(rpos)
    X2 = np.column_stack([dt / np.sqrt(rpos), dt * np.sqrt(rpos)])
    beta2, _, _, res2 = ols(y, X2, const=False)
    a_c = max(-beta2[1], 0.05)
    b_c = max(beta2[0] / a_c, 1e-3)
    s_c = float(np.std(res2)) / math.sqrt(dt)
    feller = 2 * a_c * b_c - s_c ** 2
    r0 = float(rs[-1])
    mats = np.array([0.25, 5.0, 10.0, 30.0])
    mk = np.array([float(c.px[t].dropna().iloc[-1]) / 100.0 for t in ("^IRX", "^FVX", "^TNX", "^TYX")])
    vy = [vasicek_yield(r0, a_v, b_v, s_v, T) for T in mats]
    cy = [cir_yield(r0, a_c, b_c, s_c, T) for T in mats]
    # Hull-White: theta(t) fitted to today's curve
    grid = np.linspace(0.25, 30, 600)
    yg = np.interp(grid, mats, mk)
    f0 = np.gradient(yg * grid, grid)
    th = np.gradient(f0, grid) + a_v * f0 + s_v ** 2 / (2 * a_v) * (1 - np.exp(-2 * a_v * grid))
    t1, t5 = float(np.interp(1, grid, th)), float(np.interp(5, grid, th))
    f1m = float(f0[0])
    e_r1m = f1m + s_v ** 2 / (2 * a_v ** 2) * (1 - math.exp(-a_v / 12)) ** 2
    d_r_vas = (b_v - r0) * (1 - math.exp(-a_v / 12))
    ch5 = float(c.px["^TNX"].dropna().iloc[-1] - c.px["^TNX"].dropna().iloc[-6]) * 100.0   # bp over 5d
    R.signal = clip(-0.5 * math.tanh(d_r_vas * 1e4 / 25) - 0.5 * math.tanh(ch5 / 20))
    R.headline = f"Vasicek a={a_v:.2f} b={pc(b_v)} s={pc(s_v,2)} | CIR a={a_c:.2f} b={pc(b_c)} s={pc(s_c,2)} | 10y mkt={pc(mk[2])} | E[dr 1m]={bp(d_r_vas,0)} | 10y 5d chg={ch5:+.0f}bp"
    R.lines = [
        "Vasicek:  dr = a (b - r) dt + sigma dW      exact discretisation  r_{t+1} = b(1-e^{-a dt}) + e^{-a dt} r_t + eps",
        f"   OLS:  a={a_v:.3f}  b={pc(b_v,3)}  sigma={pc(s_v,3)}  half-life={math.log(2)/a_v*12:.1f} months   r0={pc(r0,3)}",
        "   Bond: P = A(T) e^{-B(T) r},  B=(1-e^{-aT})/a,  A=exp[(b - sigma^2/2a^2)(B-T) - sigma^2 B^2/4a]",
        "CIR:      dr = a (b - r) dt + sigma sqrt(r) dW    (regress dr/sqrt(r) on dt/sqrt(r) and dt sqrt(r))",
        f"   a={a_c:.3f}  b={pc(b_c,3)}  sigma={s_c:.3f}   Feller 2ab - sigma^2 = {feller:.5f} ({'rates stay positive' if feller>0 else 'zero is reachable'})",
        "Hull-White: dr = [theta(t) - a r] dt + sigma dW,  theta(t) = df(0,t)/dt + a f(0,t) + sigma^2/(2a) (1 - e^{-2at})  (fits today's curve exactly)",
        f"   theta(1y)={pc(t1,3)}  theta(5y)={pc(t5,3)}   E[r(1m)]={pc(e_r1m,3)}",
        "Maturity     market      Vasicek      CIR",
    ] + [f"   {T:>5.2f}y     {pc(mk[i],3)}     {pc(vy[i],3)}     {pc(cy[i],3)}" for i, T in enumerate(mats)] + [
        f"Vasicek expected 1-month change in the short rate: {bp(d_r_vas,1)}   |   10y yield 5-day change: {ch5:+.1f}bp",
        "Read-across: rates grinding higher = valuation headwind for long-duration growth (QQQ); weight is small.",
    ]
    return R


# =============================================================================
# SECTION 2 -- STATISTICAL & TIME-SERIES MODELS
# =============================================================================
def _ar_fallback(y: np.ndarray, maxp=3):
    """AR(p) by OLS with AIC selection (used only when statsmodels is missing)."""
    best = None
    for p in range(1, maxp + 1):
        Y = y[p:]
        X = np.column_stack([y[p - k - 1: len(y) - k - 1] for k in range(p)])
        beta, _, _, res = ols(Y, X)
        n = len(Y)
        aic = n * math.log(res @ res / n) + 2 * (p + 2)
        if best is None or aic < best[0]:
            last = np.concatenate([[1.0], y[::-1][:p]])
            best = (aic, p, float(last @ beta))
    return best


def mod_arima(c: Ctx) -> Result:
    R = Result("arima", "ARIMA / SARIMAX", "2. Statistical & time-series", weight=1.0)
    y = c.r.iloc[-750:] * 100.0
    sd = float(y.std())
    lines = ["ARMA(p,q):  r_t = c + sum phi_i r_{t-i} + sum theta_j e_{t-j} + e_t   (d=0: returns are already stationary)"]
    if not HAVE_SM:
        aic, p, fc = _ar_fallback(y.values)
        lines.append(f"statsmodels missing -> AR({p}) by OLS (AIC={aic:.1f}); 1-step forecast = {fc:+.3f}%")
        c.info.update(arima_fc=fc, daily_sd=sd)
        R.signal = tsig(fc, 0.5 * sd)
        R.headline = f"AR({p}) fallback  forecast {fc:+.3f}%  (daily sd {sd:.2f}%)"
        R.lines = lines
        return R
    best = None
    for p in range(3):
        for q in range(3):
            try:
                m = SARIMAX(y.values, order=(p, 0, q), trend="c").fit(disp=False)
                if best is None or m.aic < best[0]:
                    best = (m.aic, p, q, m)
            except Exception:
                continue
    aic, p, q, m = best
    fc = m.get_forecast(1)
    mean1 = float(np.asarray(fc.predicted_mean)[0])
    ci = np.asarray(fc.conf_int(alpha=0.10))[0]
    names = getattr(m.model, "param_names", [f"p{i}" for i in range(len(m.params))])
    lines.append(f"Grid p,q in 0..2 by AIC -> ARIMA({p},0,{q})  AIC={aic:.1f}  params: " + ", ".join(f"{k}={v:.4f}" for k, v in zip(names, np.asarray(m.params)) if k != "sigma2"))
    lines.append(f"   1-step forecast = {mean1:+.4f}%   90% CI [{ci[0]:+.2f}%, {ci[1]:+.2f}%]   daily sd={sd:.2f}%")
    cand = [(mean1, aic, f"ARIMA({p},0,{q})")]
    try:   # SARIMAX: weekly seasonality (s=5) + lagged VIX log-change as exogenous regressor
        dv = np.log(c.px["^VIX"]).diff().reindex(y.index).shift(1).dropna()
        yy = y.loc[dv.index]
        m2 = SARIMAX(yy.values, exog=dv.values.reshape(-1, 1), order=(1, 0, 1), seasonal_order=(1, 0, 0, 5), trend="c").fit(disp=False)
        last_dv = float(np.log(c.px["^VIX"]).diff().iloc[-1])
        f2 = float(np.asarray(m2.get_forecast(1, exog=np.array([[last_dv]])).predicted_mean)[0])
        n2 = list(getattr(m2.model, "param_names", []))
        b_vix = float(np.asarray(m2.params)[n2.index("x1")]) if "x1" in n2 else float("nan")
        lines.append(f"SARIMAX(1,0,1)x(1,0,0,5) + exog dVIX:  AIC={m2.aic:.1f}  beta_VIX={b_vix:+.3f}  1-step forecast={f2:+.4f}%  (last dVIX={last_dv:+.3f})")
        cand.append((f2, m2.aic, "SARIMAX"))
    except Exception as e:
        lines.append(f"SARIMAX skipped: {e}")
    fc_best = min(cand, key=lambda t: t[1])
    dow = y.groupby(y.index.dayofweek).mean()
    today_dow = c.now.weekday()
    lines.append(f"Day-of-week mean return (3y) for today ({c.now.strftime('%a')}): {dow.get(today_dow, float('nan')):+.3f}%")
    lines.append(f"Chosen (lowest AIC): {fc_best[2]}  -> forecast {fc_best[0]:+.4f}%")
    c.info.update(arima_fc=fc_best[0], daily_sd=sd)
    R.signal = tsig(fc_best[0], 0.5 * sd)
    R.headline = f"{fc_best[2]}  forecast {fc_best[0]:+.3f}%  (daily sd {sd:.2f}%)  90% CI [{ci[0]:+.2f}%, {ci[1]:+.2f}%]"
    R.lines = lines
    return R


def gjr_fit(r: np.ndarray):
    """Gaussian quasi-ML GJR-GARCH(1,1) with variance targeting:
       h_t = w + (a + g 1[e<0]) e^2_{t-1} + b h_{t-1},   w = Var(r) (1 - a - g/2 - b)."""
    r = np.asarray(r, float) - np.mean(r)
    n, v0 = len(r), float(np.var(r))

    def path(p):
        a, g, b = p
        w = v0 * (1 - a - 0.5 * g - b)
        h = np.empty(n)
        h[0] = v0
        for t in range(1, n):
            e = r[t - 1]
            h[t] = w + (a + (g if e < 0 else 0.0)) * e * e + b * h[t - 1]
        return np.maximum(h, 1e-10), w

    def nll(p):
        if p[0] + 0.5 * p[1] + p[2] >= 0.998:
            return 1e12
        h, _ = path(p)
        return 0.5 * float(np.sum(np.log(2 * math.pi * h) + r * r / h))
    sol = optimize.minimize(nll, [0.03, 0.08, 0.88], method="L-BFGS-B", bounds=[(0, 0.4), (0, 0.6), (0, 0.995)])
    a, g, b = sol.x
    h, w = path(sol.x)
    h_next = w + (a + (g if r[-1] < 0 else 0.0)) * r[-1] ** 2 + b * h[-1]
    return dict(w=w, a=a, g=g, b=b, h=h, h_next=h_next)


def mod_garch(c: Ctx) -> Result:
    R = Result("garch", "GARCH volatility forecast", "2. Statistical & time-series", weight=1.0)
    y = c.r.iloc[-1500:] * 100.0
    if HAVE_ARCH:
        res = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
        pr = res.params
        w, a, g, b = pr["omega"], pr["alpha[1]"], pr["gamma[1]"], pr["beta[1]"]
        nu = float(pr.get("nu", float("nan")))
        fc = res.forecast(horizon=5, reindex=False)
        var5 = fc.variance.values[-1]
        mean1 = float(fc.mean.values[-1, 0])
        cond = np.asarray(res.conditional_volatility)
        engine = "arch: AR(1)-GJR-GARCH(1,1)-Student-t"
    else:
        fit = gjr_fit(y.values)
        w, a, g, b = fit["w"], fit["a"], fit["g"], fit["b"]
        nu, mean1 = float("nan"), float("nan")
        cond = np.sqrt(fit["h"])
        h1 = fit["h_next"]
        pers_ = a + 0.5 * g + b
        uncond_ = w / (1 - pers_)
        var5 = np.array([uncond_ + pers_ ** k * (h1 - uncond_) for k in range(5)])
        engine = "own QML: GJR-GARCH(1,1)-Gaussian"
    cond = cond[~np.isnan(cond)]
    pers = a + 0.5 * g + b
    uncond = w / (1 - pers) if pers < 1 else float("nan")
    half = math.log(0.5) / math.log(pers) if 0 < pers < 1 else float("nan")
    s1 = math.sqrt(var5[0]) / 100.0
    c.sig["garch"] = s1
    pctl = float((cond < math.sqrt(var5[0])).mean())
    c.info.update(garch_pctl=pctl, garch_vol=s1)
    z = (math.sqrt(var5[0]) - float(np.median(cond))) / float(np.std(cond))
    ann = s1 * math.sqrt(TD)
    last_ret = float(y.iloc[-1])
    sig = -0.30 * math.tanh(z)                                    # high-vol regimes skew bearish (leverage effect)
    if not math.isnan(mean1):
        sig += 0.7 * tsig(mean1, 0.5 * float(y.std()))
    c.info["garch_half"] = half
    R.signal = clip(sig)
    R.headline = f"{engine} | 1d vol {pc(s1,2)} ({pc(ann,1)} ann.) = {pctl*100:.0f}th pctile of history | persistence {pers:.3f} | gamma(leverage)={g:.3f}"
    R.lines = [
        "h_t = omega + (alpha + gamma 1[e_{t-1}<0]) e^2_{t-1} + beta h_{t-1}      (GJR: negative shocks add gamma)",
        f"Engine: {engine}",
        f"omega={w:.5f} alpha={a:.4f} gamma={g:.4f} beta={b:.4f}" + (f" nu(t dof)={nu:.1f}" if not math.isnan(nu) else ""),
        f"Persistence alpha+gamma/2+beta = {pers:.4f}   shock half-life = {half:.1f} days   unconditional daily vol = {pc(math.sqrt(uncond)/100 if uncond==uncond else float('nan'),2)}",
        "Forecast variance path (1..5d), as daily vol %: " + ", ".join(f"{math.sqrt(v):.2f}" for v in var5),
        f"Next-day vol = {pc(s1,3)} ({pc(ann,1)} annualised) | percentile in sample = {pctl*100:.0f}% | z vs median = {z:+.2f}",
        f"Yesterday's return {last_ret:+.2f}%: " + ("negative shock -> leverage term adds to variance" if last_ret < 0 else "positive shock -> no leverage add-on"),
    ] + ([f"AR(1) mean forecast for today: {mean1:+.4f}%"] if not math.isnan(mean1) else [])
    return R


def _adf_t(x: np.ndarray, lags=1):
    """Minimal ADF t-stat (constant, `lags` augmented terms)."""
    dx = np.diff(x)
    Y = dx[lags:]
    cols = [x[lags:-1]] + [dx[lags - k - 1: len(dx) - k - 1] for k in range(lags)]
    beta, t, _, _ = ols(Y, np.column_stack(cols))
    return float(t[1])


def mod_pairs(c: Ctx) -> Result:
    R = Result("pairs", "Engle-Granger / Johansen cointegration + Ornstein-Uhlenbeck", "2. Statistical & time-series", weight=0.3)
    eq, pr = c.cfg.equity, c.cfg.pair
    n = 504
    lq, ls = np.log(c.px[eq].iloc[-n:].values), np.log(c.px[pr].iloc[-n:].values)
    (al, be), _, _, spread = ols(lq, ls)
    lines = [f"Pair: ln({eq}) = alpha + beta ln({pr}) + spread      (last {n} days)   alpha={al:.4f}  hedge ratio beta={be:.4f}"]
    if HAVE_SM:
        t_eg, p_eg, crit = coint(lq, ls, trend="c", autolag="aic")
        adf_p = float(adfuller(spread, autolag="AIC")[1])
        lines.append(f"Engle-Granger: t={t_eg:.3f}  p={p_eg:.3f}  crit(1/5/10%)=({crit[0]:.2f},{crit[1]:.2f},{crit[2]:.2f})   ADF(spread) p={adf_p:.3f}")
        coint_ok = p_eg < 0.10
        try:
            ex = np.log(c.px["IWM"].iloc[-n:].values)
            jo = coint_johansen(np.column_stack([lq, ls, ex]), det_order=0, k_ar_diff=1)
            rank = int(sum(jo.lr1 > jo.cvt[:, 1]))
            v = jo.evec[:, 0] / jo.evec[0, 0]
            lines.append(f"Johansen ({eq},{pr},IWM): trace stats={np.round(jo.lr1,1).tolist()} vs 95% crit={np.round(jo.cvt[:,1],1).tolist()} -> cointegration rank r={rank}")
            lines.append(f"   first cointegrating vector (normalised) = {np.round(v,3).tolist()}")
        except Exception as e:
            lines.append(f"Johansen skipped: {e}")
    else:
        t_adf = _adf_t(spread, 1)
        coint_ok = t_adf < -3.34                       # approx. Engle-Granger 5% critical value (2 variables)
        lines.append(f"statsmodels missing -> own ADF on residual: t={t_adf:.2f} (5% EG crit ~ -3.34) -> {'cointegrated' if coint_ok else 'not cointegrated'}")
    # Ornstein-Uhlenbeck on the spread:  dX = kappa (mu - X) dt + sigma dW
    dt = 1 / TD
    (a0, b0), _, _, res = ols(spread[1:], spread[:-1])
    b0 = min(max(b0, 1e-3), 0.9999)
    kappa = -math.log(b0) / dt
    mu = a0 / (1 - b0)
    s_eq = float(np.std(res, ddof=2)) / math.sqrt(1 - b0 ** 2)
    sig_ou = s_eq * math.sqrt(2 * kappa)
    half = math.log(2) / kappa * TD
    x_now = math.log(c.spot) - al - be * ls[-1]
    z = (x_now - mu) / s_eq
    e1 = mu + (x_now - mu) * b0
    z1 = (e1 - mu) / s_eq
    fair = math.exp(al + be * ls[-1] + mu)          # QQQ price that puts the spread exactly at its mean
    c.info.update(pair_fair=fair, pair_z=z, pair_coint=coint_ok)
    lines += [
        "OU:  dX = kappa (mu - X) dt + sigma dW      discrete: X_{t+1} = a + b X_t + eps, b = e^{-kappa dt}, kappa = -ln(b)/dt",
        f"   kappa={kappa:.2f}/yr  mu={mu:+.4f}  sigma_OU={sig_ou:.4f}  stationary sd={s_eq:.4f}  half-life={half:.1f} trading days",
        f"   spread now (QQQ at open vs last SPY close) = {x_now:+.4f}  -> z={z:+.2f}  | E[z] tomorrow = {z1:+.2f}",
        f"   Read: z>0 means {eq} rich vs {pr} (mean-reversion pull = {eq} lags); z<0 means cheap.",
        f"   Fair value of {eq} given {pr} = exp(alpha + beta ln {pr} + mu) = {f(fair)}  vs market {f(c.spot)}  ({pc(c.spot/fair-1,2)})",
    ]
    qual = 1.0 if coint_ok else 0.3
    R.signal = clip(-0.5 * math.tanh(z / 2.0) * qual)
    R.headline = f"{'COINTEGRATED' if coint_ok else 'weak/no cointegration'} | beta={be:.3f} | OU half-life {half:.0f}d | spread z={z:+.2f}"
    R.lines = lines
    return R


def mod_factors(c: Ctx) -> Result:
    R = Result("factors", "CAPM & Fama-French multi-factor", "2. Statistical & time-series", weight=0.3)
    n = c.cfg.lookback
    rets = np.log(c.px[[c.cfg.equity, "SPY", "IWM", "IWD", "IWF", "MTUM"]]).diff().iloc[-n:]
    rf_d = (c.px["^IRX"].iloc[-n:].values / 100.0) / TD
    y = rets[c.cfg.equity].values - rf_d
    mkt = rets["SPY"].values - rf_d
    smb = (rets["IWM"] - rets["SPY"]).values
    hml = (rets["IWD"] - rets["IWF"]).values
    mom = (rets["MTUM"] - rets["SPY"]).values
    b1, t1, r2_1, _ = ols(y, mkt)
    X = np.column_stack([mkt, smb, hml, mom])
    b4, t4, r2_4, res = ols(y, X)
    names = ["alpha", "MKT", "SMB", "HML", "MOM"]
    prem = X[-60:].mean(axis=0)
    exp_ret = float(b4[1:] @ prem)
    c.info["beta"] = float(b1[1])
    sd = float(np.std(y))
    R.signal = tsig(exp_ret, 0.25 * sd)
    R.headline = f"CAPM beta={b1[1]:.2f} (R2={r2_1:.2f}) | FF4 betas MKT {b4[1]:.2f} SMB {b4[2]:+.2f} HML {b4[3]:+.2f} MOM {b4[4]:+.2f} | alpha {pc(b4[0]*TD,1)}/yr"
    R.lines = [
        f"CAPM:  R_i - R_f = alpha + beta (R_m - R_f) + e      ({n}d, proxies: MKT=SPY-rf, SMB=IWM-SPY, HML=IWD-IWF, MOM=MTUM-SPY)",
        f"   alpha={pc(b1[0]*TD,2)}/yr (t={t1[0]:.2f})  beta={b1[1]:.3f} (t={t1[1]:.1f})  R2={r2_1:.3f}",
        "FF4:   R_i - R_f = alpha + b_m MKT + b_s SMB + b_v HML + b_w MOM + e",
    ] + [f"   {nm:<5} coef={b4[i]*(TD if i==0 else 1):+.4f}{'/yr' if i==0 else ''}  t={t4[i]:+.2f}" for i, nm in enumerate(names)] + [
        f"   R2={r2_4:.3f}  residual (idiosyncratic) vol={pc(float(np.std(res))*math.sqrt(TD),1)}/yr",
        f"Factor-implied daily return from last-60d factor premia: {bp(exp_ret,2)}",
        "Interpretation: positive SMB/HML/MOM loadings = tilt toward small, value or momentum; alpha = unexplained excess return.",
    ]
    return R


def mod_markowitz(c: Ctx) -> Result:
    R = Result("markowitz", "Markowitz mean-variance / efficient frontier", "2. Statistical & time-series", weight=0.3)
    eq = c.cfg.equity
    assets = [eq, "SPY", "IWM", "TLT", "GLD"]
    rets = np.log(c.px[assets]).diff().dropna().iloc[-c.cfg.lookback:]
    mu_s = rets.mean().values * TD
    mu = 0.5 * mu_s + 0.5 * mu_s.mean()
    cov = (LedoitWolf().fit(rets.values).covariance_ if HAVE_SK else np.cov(rets.values.T)) * TD
    n, rf = len(assets), c.rf
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0, 1)] * n
    w0 = np.ones(n) / n
    vol = lambda w: math.sqrt(max(w @ cov @ w, 1e-12))
    ms = optimize.minimize(lambda w: -(w @ mu - rf) / vol(w), w0, bounds=bnds, constraints=[cons], method="SLSQP")
    mv = optimize.minimize(lambda w: w @ cov @ w, w0, bounds=bnds, constraints=[cons], method="SLSQP")
    front = []
    for tgt in np.linspace(float(mv.x @ mu), float(mu.max()), 7):
        sol = optimize.minimize(lambda w: w @ cov @ w, w0, bounds=bnds, method="SLSQP",
                                constraints=[cons, {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}])
        if sol.success:
            front.append((tgt, vol(sol.x)))
    wq = float(ms.x[0])
    R.signal = 0.5 * tsig(wq - 1.0 / n, 0.30)
    R.headline = "Max-Sharpe weights: " + ", ".join(f"{a} {100*w:.0f}%" for a, w in zip(assets, ms.x) if w > 0.005) + f" | Sharpe {(ms.x@mu-rf)/vol(ms.x):.2f}"
    R.lines = [
        "max_w (w'mu - rf)/sqrt(w' Sigma w)   s.t. sum w = 1, w >= 0   (SLSQP);   Sigma = Ledoit-Wolf shrunk covariance; mu = 50% sample / 50% grand mean",
        "Expected returns (shrunk, ann.): " + ", ".join(f"{a} {pc(m,1)}" for a, m in zip(assets, mu)),
        "Volatilities (ann.): " + ", ".join(f"{a} {pc(math.sqrt(cov[i,i]),1)}" for i, a in enumerate(assets)),
        "Max-Sharpe: " + ", ".join(f"{a} {100*w:.1f}%" for a, w in zip(assets, ms.x)) + f"   ret={pc(ms.x@mu,1)} vol={pc(vol(ms.x),1)}",
        "Min-variance: " + ", ".join(f"{a} {100*w:.1f}%" for a, w in zip(assets, mv.x)) + f"   ret={pc(mv.x@mu,1)} vol={pc(vol(mv.x),1)}",
        "Efficient frontier (ret, vol): " + "  ".join(f"({pc(a,1)}, {pc(b,1)})" for a, b in front),
        f"QQQ max-Sharpe weight {100*wq:.1f}% vs equal-weight {100/n:.0f}% -> risk-on / risk-off tilt.",
    ]
    return R


# ---------- Hidden Markov Model ----------
class _MiniHMM:
    """Gaussian HMM (diagonal covariance) trained by Baum-Welch with scaled forward-backward.  Fallback if hmmlearn is missing."""

    def __init__(self, k=3, iters=60, seed=0):
        self.k, self.iters, self.rng = k, iters, np.random.default_rng(seed)

    def _emis(self, X):
        out = np.empty((len(X), self.k))
        for j in range(self.k):
            out[:, j] = np.exp(-0.5 * np.sum(((X - self.mu[j]) ** 2) / self.var[j] + np.log(2 * np.pi * self.var[j]), axis=1))
        return np.maximum(out, 1e-300)

    def fit(self, X):
        k, T = self.k, len(X)
        order = np.argsort(X[:, 1])
        self.mu = np.array([X[order[i * T // k:(i + 1) * T // k]].mean(0) for i in range(k)])
        self.var = np.array([X[order[i * T // k:(i + 1) * T // k]].var(0) + 1e-6 for i in range(k)])
        self.A = np.full((k, k), 0.05 / (k - 1)) + np.eye(k) * (0.95 - 0.05 / (k - 1))
        self.pi = np.ones(k) / k
        ll_old = -np.inf
        ll = ll_old
        for _ in range(self.iters):
            B = self._emis(X)
            al, sc = np.zeros((T, k)), np.zeros(T)
            al[0] = self.pi * B[0]
            sc[0] = al[0].sum(); al[0] /= sc[0]
            for t in range(1, T):
                al[t] = (al[t - 1] @ self.A) * B[t]
                sc[t] = al[t].sum(); al[t] /= sc[t]
            be = np.ones((T, k))
            for t in range(T - 2, -1, -1):
                be[t] = (self.A @ (B[t + 1] * be[t + 1])) / sc[t + 1]
            gam = al * be
            gam /= gam.sum(1, keepdims=True)
            xi = np.zeros((k, k))
            for t in range(T - 1):
                m = al[t][:, None] * self.A * (B[t + 1] * be[t + 1])[None, :]
                xi += m / m.sum()
            self.A = xi / xi.sum(1, keepdims=True)
            self.pi = gam[0]
            for j in range(k):
                w = gam[:, j][:, None]
                self.mu[j] = (w * X).sum(0) / w.sum()
                self.var[j] = (w * (X - self.mu[j]) ** 2).sum(0) / w.sum() + 1e-8
            ll = float(np.log(sc).sum())
            if abs(ll - ll_old) < 1e-4:
                break
            ll_old = ll
        self.ll, self.last_post = ll, al[-1]
        return self


def mod_hmm(c: Ctx) -> Result:
    R = Result("hmm", "Hidden Markov regime model", "2. Statistical & time-series", weight=1.5)
    r = c.r.iloc[-1500:]
    rv5 = np.log(r.rolling(5).std() + 1e-6)
    d = pd.DataFrame({"r": r * 100.0, "lv": rv5}).dropna()
    X = d.values
    K = 3
    if HAVE_HMM:
        best = None
        for sd_ in range(5):
            try:
                m = GaussianHMM(K, covariance_type="diag", n_iter=200, random_state=sd_).fit(X)
                ll = m.score(X)
                if best is None or ll > best[0]:
                    best = (ll, m)
            except Exception:
                continue
        ll, m = best
        mu, A = m.means_, m.transmat_
        post = m.predict_proba(X)[-1]
        var = m.covars_ if m.covars_.ndim == 2 else np.array([np.diag(cv) for cv in m.covars_])
        engine = "hmmlearn GaussianHMM"
    else:
        best = None
        for sd_ in range(3):
            m = _MiniHMM(K, seed=sd_).fit(X)
            if best is None or m.ll > best.ll:
                best = m
        m = best
        mu, A, post, var, ll = m.mu, m.A, m.last_post, m.var, m.ll
        engine = "own Baum-Welch HMM"
    order = np.argsort(mu[:, 1])                            # sort states by log-vol feature
    lab = {int(order[0]): "CALM/BULL", int(order[1]): "NORMAL", int(order[2]): "STRESS/HIGH-VOL"}
    if mu[order[0], 0] < mu[order[1], 0] - 0.05:
        lab[int(order[0])], lab[int(order[1])] = "CALM/CHOP", "BULL"
    p_next = post @ A
    e_ret = float(p_next @ mu[:, 0]) / 100.0
    s = int(np.argmax(post))
    c.info.update(hmm_state=lab[s], hmm_prob=float(post[s]))
    sd_state = [math.sqrt(var[j][0]) for j in range(K)]
    R.signal = tsig(e_ret, 0.5 * float(r.std()))
    R.headline = f"{engine} | state NOW: {lab[s]} ({post[s]*100:.0f}%) | next-day E[ret]={pc(e_ret,3)} | P(stay)={A[s,s]*100:.0f}%"
    R.lines = [
        "r_t | S_t=j ~ N(mu_j, sigma_j^2);  P(S_{t+1}=j | S_t=i) = A_ij;   filtered posterior via forward algorithm; 3 states on [return %, ln 5d vol]",
        f"Engine: {engine}   log-likelihood={ll:.1f}   obs={len(X)}",
    ]
    for j in order:
        R.lines.append(f"   state {lab[int(j)]:<16} mean ret={mu[j,0]:+.3f}%/d  sd={sd_state[j]:.2f}%  exp.duration={1/(1-A[j,j]):.1f}d  posterior now={post[j]*100:.1f}%  next-day prob={p_next[j]*100:.1f}%")
    R.lines.append("Transition matrix rows (from) -> cols (to), ordered calm/normal/stress:")
    for i in order:
        R.lines.append("   " + "  ".join(f"{A[i,j]:.3f}" for j in order))
    R.lines.append(f"Next-day expected return = sum_j P(S_t+1=j) mu_j = {pc(e_ret,3)}")
    return R


# =============================================================================
# SECTION 3 -- NUMERICAL & COMPUTATIONAL METHODS
# =============================================================================
def geo_asian_call(S, K, T, r, q, sig, n):
    """Closed form for a discretely monitored GEOMETRIC Asian call (used as control variate)."""
    mu_g = math.log(S) + (r - q - 0.5 * sig ** 2) * T * (n + 1) / (2 * n)
    var_g = sig ** 2 * T * (n + 1) * (2 * n + 1) / (6 * n ** 2)
    sg = math.sqrt(var_g)
    d1 = (mu_g + var_g - math.log(K)) / sg
    return math.exp(-r * T) * (math.exp(mu_g + 0.5 * var_g) * Ncdf(d1) - K * Ncdf(d1 - sg))


def mod_montecarlo(c: Ctx) -> Result:
    R = Result("mc", "Monte Carlo + variance reduction", "3. Numerical & computational", weight=0.0)
    S, r, q, sig = c.spot, c.rf, c.q, c.atm_iv
    T, K, N = 30 / 365, c.spot, c.cfg.mc_paths
    rng = np.random.default_rng(c.cfg.seed)
    disc = math.exp(-r * T)
    exact = bs_price(S, K, T, r, q, sig, "c")
    mu_rn = (r - q - 0.5 * sig ** 2) * T
    sv = sig * math.sqrt(T)
    Z = rng.standard_normal(N)
    ST = S * np.exp(mu_rn + sv * Z)
    pay = disc * np.maximum(ST - K, 0)
    plain, se_plain = pay.mean(), pay.std(ddof=1) / math.sqrt(N)
    ST2 = S * np.exp(mu_rn - sv * Z)
    pair = 0.5 * (pay + disc * np.maximum(ST2 - K, 0))
    anti, se_anti = pair.mean(), pair.std(ddof=1) / math.sqrt(N)
    ctrl = disc * (ST - S * math.exp((r - q) * T))
    beta = np.cov(pay, ctrl)[0, 1] / ctrl.var(ddof=1)
    cvp = pay - beta * ctrl
    cv, se_cv = cvp.mean(), cvp.std(ddof=1) / math.sqrt(N)
    lines = [
        "Estimator: V = e^{-rT} E[(S_T-K)+],  S_T = S exp((r-q-sigma^2/2)T + sigma sqrt(T) Z);   Var(plain) = Var(payoff)/N",
        f"30d ATM call (K={f(K)}, IV={pc(sig,1)}):  Black-Scholes exact = {exact:.4f}",
        f"   plain MC      {plain:.4f}  se={se_plain:.5f}",
        f"   antithetic    {anti:.4f}  se={se_anti:.5f}   variance reduction x{(se_plain/se_anti)**2:.1f}",
        f"   control var.  {cv:.4f}  se={se_cv:.5f}   variance reduction x{(se_plain/se_cv)**2:.1f}   (beta*={beta:.3f}, control = discounted S_T)",
    ]
    s1 = max(c.sig.get("garch", c.sig.get("gbm", 0.012)), 1e-4)
    m1 = -0.5 * s1 ** 2
    lines.append(f"Importance sampling, 1-day tail P(return < -x) with sigma={pc(s1,2)}:  tilt Z~N(theta,1), weight exp(-theta Z + theta^2/2), theta=(ln(1-x)-m)/s")
    tails = {}
    for x in (0.01, 0.02, 0.03):
        L = math.log(1 - x)
        th = (L - m1) / s1
        Zs = rng.standard_normal(N) + th
        w = np.exp(-th * Zs + 0.5 * th ** 2)
        est = w * ((m1 + s1 * Zs) < L)
        p_is, se_is = est.mean(), est.std(ddof=1) / math.sqrt(N)
        ind = ((m1 + s1 * rng.standard_normal(N)) < L).astype(float)
        p_pl, se_pl = ind.mean(), ind.std(ddof=1) / math.sqrt(N)
        p_ex = float(Ncdf((L - m1) / s1))
        tails[x] = p_ex
        lines.append(f"   x={100*x:.0f}%: exact={p_ex:.3e}  IS={p_is:.3e} (se {se_is:.1e})  plain={p_pl:.3e} (se {se_pl:.1e})  IS variance gain ~x{(se_pl/max(se_is,1e-30))**2:.0f}")
    nu = 5.0
    tt = rng.standard_t(nu, N) * math.sqrt((nu - 2) / nu)
    rr = m1 + s1 * tt
    var95, var99 = -np.quantile(rr, 0.05), -np.quantile(rr, 0.01)
    es95 = -rr[rr <= -var95].mean()
    c.info.update(tail1=tails[0.01], tail2=tails[0.02], var95=float(var95), es95=float(es95))
    nq_var = 20 * S * c.fut_ratio * var95 if c.fut_ratio == c.fut_ratio else float("nan")
    lines.append(f"1-day VaR/ES (Student-t nu=5): VaR95={pc(var95,2)}  VaR99={pc(var99,2)}  ES95={pc(es95,2)}   (QQQ ${S*var95:.2f} / NQ contract ${nq_var:,.0f} @95%)")
    n_obs, Np = 21, 100_000
    dtm = T / n_obs
    Zp = rng.standard_normal((Np, n_obs))
    paths = S * np.exp(np.cumsum((r - q - 0.5 * sig ** 2) * dtm + sig * math.sqrt(dtm) * Zp, axis=1))
    arith = disc * np.maximum(paths.mean(1) - K, 0)
    geo = disc * np.maximum(np.exp(np.log(paths).mean(1)) - K, 0)
    geo_ex = geo_asian_call(S, K, T, r, q, sig, n_obs)
    b2 = np.cov(arith, geo)[0, 1] / geo.var(ddof=1)
    cva = arith - b2 * (geo - geo_ex)
    lines.append(f"Arithmetic Asian call (21 daily fixings, no closed form): plain={arith.mean():.4f} (se {arith.std(ddof=1)/math.sqrt(Np):.5f})  "
                 f"with geometric-Asian control variate={cva.mean():.4f} (se {cva.std(ddof=1)/math.sqrt(Np):.5f})  variance gain x{(arith.std()/cva.std())**2:.0f}")
    R.headline = f"IS tails P(-1%)={tails[0.01]*100:.1f}% P(-2%)={tails[0.02]*100:.2f}% P(-3%)={tails[0.03]*100:.3f}% | t5 VaR95={pc(var95,2)} ES95={pc(es95,2)} | CV gain x{(se_plain/se_cv)**2:.0f}"
    R.lines = lines
    return R


def fd_price(S0, K, T, r, q, sig, kind="p", american=False, theta=0.5, M=300, N=300):
    """Finite differences for the BS PDE in tau = T - t:  V_tau = 1/2 s^2 S^2 V_SS + (r-q) S V_S - r V.
    theta = 0 explicit, 1 implicit, 0.5 Crank-Nicolson.  American = projection onto the payoff each step."""
    Smax = max(S0, K) * max(1.6, 1.0 + 7.0 * sig * math.sqrt(T))
    m0 = max(1, int(round(S0 / (Smax / M))))
    ds = S0 / m0                                           # S0 sits exactly on a grid node
    Sg = ds * np.arange(M + 1)
    if theta == 0.0:
        N = max(N, int(T * sig ** 2 * M * M * 1.6) + 1)       # explicit scheme stability
    dt = T / N
    i = np.arange(1, M)
    a = 0.5 * dt * (sig ** 2 * i ** 2 - (r - q) * i)
    b = -dt * (sig ** 2 * i ** 2 + r)
    cc = 0.5 * dt * (sig ** 2 * i ** 2 + (r - q) * i)
    pay = np.maximum(Sg - K, 0) if kind == "c" else np.maximum(K - Sg, 0)
    V = pay.copy()
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = -theta * cc[:-1]
    ab[1, :] = 1 - theta * b
    ab[2, :-1] = -theta * a[1:]
    for n in range(1, N + 1):
        tau = n * dt
        if kind == "p":
            lo, hi = K * math.exp(-r * tau), 0.0
            if american:
                lo = K
        else:
            lo, hi = 0.0, Sg[-1] * math.exp(-q * tau) - K * math.exp(-r * tau)
        Vi = V[1:M]
        rhs = Vi + (1 - theta) * (a * V[0:M - 1] + b * Vi + cc * V[2:M + 1])
        rhs[0] += theta * a[0] * lo
        rhs[-1] += theta * cc[-1] * hi
        new = solve_banded((1, 1), ab, rhs) if theta > 0 else rhs
        V[0], V[M], V[1:M] = lo, hi, new
        if american:
            V = np.maximum(V, pay)
    price = float(np.interp(S0, Sg, V))
    g = np.gradient(V, ds)
    gg = np.gradient(g, ds)
    return price, float(np.interp(S0, Sg, g)), float(np.interp(S0, Sg, gg)), N


def mod_fdm(c: Ctx) -> Result:
    R = Result("fdm", "Finite-difference PDE solvers", "3. Numerical & computational", weight=0.0)
    S, r, q, sig, T = c.spot, c.rf, c.q, c.atm_iv, 30 / 365
    K = S
    bs = bs_price(S, K, T, r, q, sig, "p")
    gk = bs_greeks(S, K, T, r, q, sig, "p")
    lines = [
        "PDE:  V_tau = 1/2 sigma^2 S^2 V_SS + (r-q) S V_S - r V ;  theta-scheme (I - theta L) V^{n+1} = (I + (1-theta) L) V^n, tridiagonal solve (banded LU)",
        f"European ATM put (K={f(K)}, T=30d, IV={pc(sig,1)}):  Black-Scholes = {bs:.5f}   delta={gk['delta']:+.4f} gamma={gk['gamma']:.5f}",
    ]
    cn_err = float("nan")
    for nm, th in (("explicit", 0.0), ("implicit", 1.0), ("Crank-Nicolson", 0.5)):
        p, d, g, nn = fd_price(S, K, T, r, q, sig, "p", False, th, 200, 200)
        if th == 0.5:
            cn_err = p - bs
        lines.append(f"   {nm:<15} price={p:.5f}  err={p-bs:+.2e}  delta={d:+.4f}  gamma={g:.5f}   (steps used {nn})")
    lines.append("Convergence of Crank-Nicolson (grid M x N): " + "  ".join(
        f"{m}x{m}: err {fd_price(S, K, T, r, q, sig, 'p', False, 0.5, m, m)[0]-bs:+.1e}" for m in (50, 100, 200, 400)))
    Ki = 1.05 * S
    eu = bs_price(S, Ki, T, r, q, sig, "p")
    am, d_am, g_am, _ = fd_price(S, Ki, T, r, q, sig, "p", True, 0.5, 300, 300)
    lines.append(f"American put, K={f(Ki)} (5% ITM): European={eu:.4f}  American (CN + projection)={am:.4f}  early-exercise premium={am-eu:.4f}  delta={d_am:+.4f}")
    c.info["cn_err"] = cn_err
    R.headline = f"CN err {cn_err:+.1e} vs BS | American 5%-ITM put {am:.3f} (early-ex premium {am-eu:.3f})"
    R.lines = lines
    return R


def mod_rootfind(c: Ctx) -> Result:
    R = Result("root", "Root-finding: Newton-Raphson & Brent (implied vol)", "3. Numerical & computational", weight=0.0)
    S, r, q = c.spot, c.rf, c.q
    lines = ["Solve BS(sigma) - market = 0.   Newton: s_{n+1} = s_n - (BS(s_n)-P)/vega(s_n) (quadratic);  Brent: bisection + secant + inverse-quadratic (guaranteed, super-linear)"]
    ch = c.near or c.term
    if ch is not None and c.smile_near is not None and len(c.smile_near):
        T, K = ch["T"], c.atm_strike
        px_ = bs_price(S, K, T, r, q, c.atm_iv, "c")
        nw, nn = iv_newton(px_, S, K, T, r, q, "c")
        br, nb = iv_brent(px_, S, K, T, r, q, "c")
        bi, ni = iv_bisect(px_, S, K, T, r, q, "c")
        lines.append(f"ATM call {px_:.4f}:  Newton iv={nw:.8f} in {nn} it | Brent iv={br:.8f} in {nb} it | bisection iv={bi:.8f} in {ni} it")
        s, errs = 0.60, []
        for _ in range(7):
            errs.append(abs(s - c.atm_iv))
            v = bs_vega_raw(S, K, T, r, q, s)
            if v < 1e-12:
                break
            s = min(max(s - (bs_price(S, K, T, r, q, s, "c") - px_) / v, 1e-4), 8.0)
        lines.append("Newton error |s_n - s*| from a bad start (60%): " + ", ".join(f"{e:.1e}" for e in errs) + "  (digits roughly double per step)")
        sm = c.smile_near
        nw_fail, diffs = 0, []
        for _, o in sm.iterrows():
            a1, _ = iv_newton(o["mid"], S, o["K"], T, r, q, o["kind"])
            if math.isnan(a1):
                nw_fail += 1
            else:
                diffs.append(abs(a1 - o["iv"]))
        lines.append(f"Smile ({len(sm)} OTM strikes): IV range {pc(sm['iv'].min(),1)} - {pc(sm['iv'].max(),1)};  Newton failed on {nw_fail};  max |Newton-Brent|={max(diffs) if diffs else float('nan'):.1e}")
        R.headline = f"Newton {nn} it vs Brent {nb} it vs bisection {ni} it | smile {len(sm)} strikes {pc(sm['iv'].min(),0)}-{pc(sm['iv'].max(),0)}"
    else:
        T, K = 30 / 365, S
        px_ = bs_price(S, K, T, r, q, 0.22, "c")
        nw, nn = iv_newton(px_, S, K, T, r, q, "c")
        br, nb = iv_brent(px_, S, K, T, r, q, "c")
        lines.append(f"(no chain) synthetic test, true 22%: Newton {nw:.6f} ({nn} it), Brent {br:.6f} ({nb} it)")
        R.headline = f"Newton {nn} it, Brent {nb} it (no live chain)"
    R.lines = lines
    return R


def crr_tree(S, K, T, r, q, sig, kind="p", american=True, N=500):
    dt = T / N
    u = math.exp(sig * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(N + 1)
    ST = S * u ** (N - j) * d ** j
    V = np.maximum(ST - K, 0) if kind == "c" else np.maximum(K - ST, 0)
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:i + 1] + (1 - p) * V[1:i + 2])
        if american:
            jj = np.arange(i + 1)
            Si = S * u ** (i - jj) * d ** jj
            V = np.maximum(V, np.maximum(Si - K, 0) if kind == "c" else np.maximum(K - Si, 0))
    return float(V[0])


def trinomial_tree(S, K, T, r, q, sig, kind="p", american=True, N=300):
    dt = T / N
    dx = sig * math.sqrt(3 * dt)
    nu = r - q - 0.5 * sig ** 2
    A = (sig ** 2 * dt + nu ** 2 * dt ** 2) / dx ** 2
    pu, pd_ = 0.5 * (A + nu * dt / dx), 0.5 * (A - nu * dt / dx)
    pm = 1 - pu - pd_
    disc = math.exp(-r * dt)
    ST = S * np.exp(np.arange(-N, N + 1) * dx)
    V = np.maximum(ST - K, 0) if kind == "c" else np.maximum(K - ST, 0)
    for step in range(N - 1, -1, -1):
        V = disc * (pu * V[2:] + pm * V[1:-1] + pd_ * V[:-2])
        if american:
            Si = S * np.exp(np.arange(-step, step + 1) * dx)
            V = np.maximum(V, np.maximum(Si - K, 0) if kind == "c" else np.maximum(K - Si, 0))
    return float(V[0])


def mod_lattice(c: Ctx) -> Result:
    R = Result("lattice", "Binomial (CRR) & trinomial lattices", "3. Numerical & computational", weight=0.0)
    S, r, q, sig, T = c.spot, c.rf, c.q, c.atm_iv, 30 / 365
    K = 1.05 * S
    bs = bs_price(S, K, T, r, q, sig, "p")
    lines = [
        "CRR: u=e^{sigma sqrt(dt)}, d=1/u, p=(e^{(r-q)dt}-d)/(u-d);  Trinomial: dx=sigma sqrt(3dt), p_u,p_d = 1/2[(sigma^2 dt + nu^2 dt^2)/dx^2 +/- nu dt/dx], nu=r-q-sigma^2/2",
        f"5% ITM put K={f(K)} T=30d IV={pc(sig,1)}:  BS European = {bs:.5f}",
        "   N      CRR-Euro   err         CRR-Amer   Trinom-Amer",
    ]
    for N in (25, 50, 100, 250, 500):
        ce = crr_tree(S, K, T, r, q, sig, "p", False, N)
        ca = crr_tree(S, K, T, r, q, sig, "p", True, N)
        ta = trinomial_tree(S, K, T, r, q, sig, "p", True, min(N, 250))
        lines.append(f"   {N:<5}  {ce:.5f}   {ce-bs:+.1e}    {ca:.5f}    {ta:.5f}")
    am = crr_tree(S, K, T, r, q, sig, "p", True, 500)
    ta250 = trinomial_tree(S, K, T, r, q, sig, "p", True, 250)
    lines.append(f"Early-exercise premium (CRR N=500): {am-bs:.5f}   (early-exercise payoffs are easy on a lattice: take max(continuation, intrinsic) at every node)")
    R.headline = f"CRR(500) American put {am:.4f} vs BS Euro {bs:.4f} | trinomial agrees to {abs(ta250-am):.1e}"
    R.lines = lines
    return R


# =============================================================================
# SECTION 4 -- MACHINE LEARNING & EXECUTION MODELS
# =============================================================================
def _hawkes2_fit(t: np.ndarray, m: np.ndarray, T: float):
    """Bivariate exponential Hawkes (mark 0 = up-shock, 1 = down-shock), MLE.
       lambda_k(t) = mu_k + sum_l sum_{t_j<t, m_j=l} alpha_kl * beta * exp(-beta (t - t_j))."""
    n = len(t)

    def unpack(th):
        return np.exp(th[0:2]), np.exp(th[2:6]).reshape(2, 2), math.exp(th[6])

    def nll(th):
        mu, al, be = unpack(th)
        if max(abs(np.linalg.eigvals(al))) >= 0.995:
            return 1e9
        A = np.zeros(2)
        ll = 0.0
        for i in range(n):
            if i > 0:
                A *= math.exp(-be * (t[i] - t[i - 1]))
            k = m[i]
            lam = mu[k] + be * float(al[k] @ A)
            ll += math.log(max(lam, 1e-12))
            A[k] += 1.0
        comp = float(mu.sum() * T)
        for i in range(n):
            comp += float(al[:, m[i]].sum()) * (1 - math.exp(-be * (T - t[i])))
        return -(ll - comp)
    rate = n / T
    x0 = np.log([rate / 2, rate / 2, 0.15, 0.05, 0.05, 0.15, 0.1])
    bnds = [(-12, 0)] * 2 + [(-9, 0)] * 4 + [(-5, 1)]
    sol = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=200))
    mu, al, be = unpack(sol.x)
    return mu, al, be, -float(sol.fun)


def mod_hawkes(c: Ctx) -> Result:
    R = Result("hawkes", "Hawkes self-exciting point process", "4. Machine learning & execution", weight=0.7)
    df = c.intra
    if df is None or len(df) < 600:
        raise RuntimeError("need >= 600 five-minute bars of history")
    ret = np.log(df["close"] / df["open"]).dropna().values
    sd = float(np.std(ret))
    thr = 2.0 * sd
    idx = np.where(np.abs(ret) > thr)[0]
    if len(idx) < 25:
        raise RuntimeError(f"only {len(idx)} shock events; not enough to fit")
    t = idx * 5.0                                        # trading-clock minutes (overnight gaps removed)
    mk = (ret[idx] < 0).astype(int)                      # 0 up, 1 down
    T = len(ret) * 5.0
    mu, al, be, ll = _hawkes2_fit(t, mk, T)
    rho_sp = float(max(abs(np.linalg.eigvals(al))))
    lam_bar = np.linalg.solve(np.eye(2) - al, mu)        # stationary mean intensity
    A = np.zeros(2)
    for i in range(len(t)):
        A[mk[i]] += math.exp(-be * (T - t[i]))
    lam_T = mu + be * (al @ A)
    ex = (lam_T - lam_bar) / lam_bar
    net = float(ex[0] - ex[1])
    c.info.update(hawkes_rho=rho_sp, hawkes_thr=thr, hawkes_ex_up=float(ex[0]), hawkes_ex_dn=float(ex[1]), hawkes_mem=1 / be)
    R.signal = clip(0.5 * math.tanh(net))
    n_up, n_dn = int((mk == 0).sum()), int((mk == 1).sum())
    R.headline = (f"{len(idx)} shocks (>2sd 5m bars: {n_up} up / {n_dn} down) | branching ratio rho={rho_sp:.2f} | memory 1/beta={1/be:.0f}min | "
                  f"excess intensity up {ex[0]*100:+.0f}% / down {ex[1]*100:+.0f}%")
    R.lines = [
        "lambda_k(t) = mu_k + sum_l sum_{t_j<t, m_j=l} alpha_kl beta e^{-beta (t-t_j)};  log L = sum ln lambda_{m_i}(t_i) - sum_k mu_k T - sum_i sum_k alpha_{k m_i}(1 - e^{-beta(T-t_i)})",
        f"Data: {len(ret)} five-minute bars, shock = |bar return| > 2 sd = {pc(thr,3)};  trading-clock time, T={T:.0f} min",
        f"MLE: mu_up={mu[0]:.5f} mu_down={mu[1]:.5f} per min | beta={be:.4f}/min (e-folding {1/be:.1f} min) | log-lik={ll:.1f}",
        f"alpha matrix (row=affected, col=trigger):  up<-up {al[0,0]:.3f}  up<-down {al[0,1]:.3f}  down<-up {al[1,0]:.3f}  down<-down {al[1,1]:.3f}",
        f"Spectral radius (branching ratio) = {rho_sp:.3f}  (<1 stationary; closer to 1 = more of the activity is self-generated)",
        f"Stationary intensity: up {lam_bar[0]*60:.2f}/h  down {lam_bar[1]*60:.2f}/h   |  intensity at last close: up {lam_T[0]*60:.2f}/h  down {lam_T[1]*60:.2f}/h",
        f"Excess over stationary at last close: up {ex[0]*100:+.0f}%  down {ex[1]*100:+.0f}%   -> net (up-down) = {net:+.2f}  (carried into the open at 50% weight)",
    ]
    return R


def mod_avellaneda(c: Ctx) -> Result:
    R = Result("as", "Avellaneda-Stoikov market-making", "4. Machine learning & execution", weight=0.3)
    S = c.spot
    s_day = c.sig.get("garch", c.sig.get("gbm", 0.012))
    sig_px = S * s_day                                    # $ per sqrt(session)
    g, k = c.cfg.as_gamma, c.cfg.as_k
    df = c.intra
    q_hat, z_imb = 0.0, 0.0
    if df is not None and len(df) > 200:
        sign = np.sign(np.log(df["close"] / df["open"]).values)
        sv = sign * df["volume"].values
        roll = pd.Series(sv).rolling(12).sum().dropna()
        z_imb = float((roll.iloc[-1] - roll.mean()) / (roll.std() + 1e-9))
        q_hat = -clip(z_imb, -3, 3)                       # aggressive buying leaves the maker SHORT (q<0)
    tau = 1.0                                             # remaining session fraction at the open
    res_shift = -q_hat * g * sig_px ** 2 * tau            # r - s
    delta = g * sig_px ** 2 * tau + (2 / g) * math.log(1 + g / k)
    skew = res_shift / (delta / 2)
    c.info.update(as_res=S + res_shift, as_bid=S + res_shift - delta / 2, as_ask=S + res_shift + delta / 2, as_z=z_imb, as_q=q_hat)
    R.signal = clip(skew / 3.0, -1, 1) * 0.5
    R.headline = f"flow imbalance z={z_imb:+.2f} -> maker inventory q={q_hat:+.2f} | reservation shift {res_shift:+.4f}$ | optimal spread ${delta:.3f} | quote skew {skew:+.2f} half-spreads"
    R.lines = [
        "Reservation price:  r(s,q,t) = s - q gamma sigma^2 (T - t)",
        "Optimal total spread:  delta^a + delta^b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)",
        f"Inputs: s={f(S)}  sigma={pc(s_day,2)}/session = ${sig_px:.2f}  gamma={g}  k={k}  (T-t)=1 session",
        f"Order-flow proxy (signed volume, last 12 bars of prior session) z={z_imb:+.2f}  -> assumed maker inventory q={q_hat:+.2f} units",
        f"r - s = {res_shift:+.4f}$  |  optimal spread = ${delta:.4f}  |  bid = r - delta/2 = {f(S+res_shift-delta/2,3)}  ask = r + delta/2 = {f(S+res_shift+delta/2,3)}",
        "Inventory ladder (q: reservation shift $):  " + "  ".join(f"{qq:+d}:{-qq*g*sig_px**2*tau:+.3f}" for qq in (-4, -2, 0, 2, 4)),
        "Spread vs time left (session fractions 1.0 / 0.5 / 0.1):  " + "  ".join(f"${g*sig_px**2*tt+(2/g)*math.log(1+g/k):.3f}" for tt in (1.0, 0.5, 0.1)),
        "Read: a maker who is short (after aggressive buying) lifts both quotes -> mild upward pressure; long inventory does the reverse.",
    ]
    return R


def _rsi(cl: pd.Series, n=14) -> pd.Series:
    d = cl.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def build_features(c: Ctx):
    """Everything known at TODAY's open  ->  target: open-to-close log return of that day."""
    o = c.ohlc
    cl, op, hi, lo, vol = o["close"], o["open"], o["high"], o["low"], o["volume"]
    px = c.px.reindex(o.index).ffill()
    r1 = np.log(cl).diff()
    G = pd.DataFrame(index=o.index)
    G["ret1"], G["ret2"], G["ret3"] = r1, r1.shift(1), r1.shift(2)
    for n in (5, 10, 21):
        G[f"ret{n}"] = np.log(cl / cl.shift(n))
    G["rv5"], G["rv21"] = r1.rolling(5).std(), r1.rolling(21).std()
    G["rvr"] = G["rv5"] / (G["rv21"] + 1e-9)
    G["rsi"] = _rsi(cl) / 100 - 0.5
    G["dma20"], G["dma50"] = np.log(cl / cl.rolling(20).mean()), np.log(cl / cl.rolling(50).mean())
    G["rng"] = (hi - lo) / cl
    lv = np.log(vol + 1)
    G["volz"] = (lv - lv.rolling(60).mean()) / (lv.rolling(60).std() + 1e-9)
    vix = np.log(px["^VIX"])
    G["vix"], G["dvix"], G["vixma"] = vix, vix.diff(), vix - vix.rolling(20).mean()
    G["d10y5"], G["slope"] = px["^TNX"].diff(5), px["^TNX"] - px["^IRX"]
    G["rel5"] = np.log(px[c.cfg.pair] / px[c.cfg.pair].shift(5)) - G["ret5"]
    G["tlt5"] = np.log(px["TLT"] / px["TLT"].shift(5))
    G["gld5"] = np.log(px["GLD"] / px["GLD"].shift(5))
    G["oc"] = np.log(cl / op)
    F = G.shift(1)
    F["gap"] = np.log(op / cl.shift(1))
    for dd in range(5):
        F[f"dow{dd}"] = (o.index.dayofweek == dd).astype(float)
    y = np.log(cl / op)
    data = pd.concat([F, y.rename("y")], axis=1).dropna().iloc[-1200:]
    live = G.iloc[[-1]].copy()
    live["gap"] = math.log(c.spot / float(cl.iloc[-1]))
    for dd in range(5):
        live[f"dow{dd}"] = float(c.now.weekday() == dd)
    live = live[F.columns]
    return data.drop(columns="y"), data["y"], live


def _skill(hit, ic, auc=None):
    q = 0.5 * clip((hit - 0.5) / 0.05, 0, 1)
    q += 0.5 * (clip((auc - 0.5) / 0.08, 0, 1) if auc is not None else clip(ic / 0.10, 0, 1))
    return q


def mod_linear_ml(c: Ctx) -> Result:
    R = Result("linear", "Lasso / Ridge / ElasticNet regression", "4. Machine learning & execution", weight=1.0)
    if not HAVE_SK:
        raise RuntimeError("scikit-learn missing")
    X, y, live = build_features(c)
    yp = y * 100.0
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=5, test_size=max(40, n // 10))
    makers = {
        "Lasso": lambda: LassoCV(cv=TimeSeriesSplit(3), n_alphas=40, max_iter=20000),
        "Ridge": lambda: RidgeCV(alphas=np.logspace(-1, 4, 30)),
        "ElasticNet": lambda: ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=TimeSeriesSplit(3), n_alphas=30, max_iter=20000),
    }
    sd_oc = float(yp.std())
    sigs, preds_today, hits = [], {}, []
    lines = [
        "min (1/2n)||y - Xb||^2 + alpha [ l1 |b|_1 + (1-l1)/2 |b|_2^2 ]   Lasso: l1=1 (sparse) | Ridge: l1=0 (shrink) | ElasticNet: mix; alpha by time-series CV",
        f"{X.shape[1]} features x {n} days; target = today's open-to-close return (%), features known at the open; walk-forward OOS ({tscv.n_splits} folds)",
    ]
    for name, mk in makers.items():
        pr = np.full(n, np.nan)
        for tr, te in tscv.split(X):
            sc = StandardScaler().fit(X.iloc[tr])
            m = mk().fit(sc.transform(X.iloc[tr]), yp.iloc[tr])
            pr[te] = m.predict(sc.transform(X.iloc[te]))
        ok = ~np.isnan(pr)
        hit = float((np.sign(pr[ok]) == np.sign(yp.values[ok])).mean())
        ic = float(np.corrcoef(pr[ok], yp.values[ok])[0, 1]) if np.std(pr[ok]) > 0 else 0.0
        r2 = 1 - float(((yp.values[ok] - pr[ok]) ** 2).sum() / (yp.values[ok] ** 2).sum())
        sc = StandardScaler().fit(X)
        m = mk().fit(sc.transform(X), yp)
        p_today = float(m.predict(sc.transform(live))[0])
        coefs = pd.Series(m.coef_, index=X.columns)
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:5]
        q = _skill(hit, ic)
        sigs.append(q * tsig(p_today, 0.5 * sd_oc))
        preds_today[name] = p_today
        hits.append(hit)
        lines.append(f"{name:<10} OOS hit-rate={hit*100:.1f}%  IC={ic:+.3f}  R2(vs 0)={r2:+.4f}  skill-weight={q:.2f}  today's pred={p_today:+.3f}%  nonzero coefs={(coefs.abs()>1e-9).sum()}"
                     + (f"  alpha={m.alpha_:.4g}" if hasattr(m, "alpha_") else ""))
        lines.append("           top features: " + ", ".join(f"{k}={v:+.3f}" for k, v in top.items()))
    c.info.update(reg_pred=float(np.mean(list(preds_today.values()))), reg_hit=float(np.mean(hits)))
    R.signal = clip(float(np.mean(sigs)))
    R.headline = " | ".join(f"{k} {v:+.2f}%" for k, v in preds_today.items()) + f" (open->close, daily sd {sd_oc:.2f}%)"
    lines.append("Skill-weight shrinks each model's vote by its out-of-sample hit-rate / IC; near-chance models count for ~0.")
    R.lines = lines
    return R


def mod_trees(c: Ctx) -> Result:
    R = Result("trees", "Random forest / gradient boosting", "4. Machine learning & execution", weight=1.0)
    if not HAVE_SK:
        raise RuntimeError("scikit-learn missing")
    X, y, live = build_features(c)
    up = (y > 0).astype(int)
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=5, test_size=max(40, n // 10))
    makers = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=20, n_jobs=-1, random_state=c.cfg.seed),
        "HistGB": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=c.cfg.seed),
    }
    if lgb is not None:
        makers["LightGBM"] = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, verbose=-1, random_state=c.cfg.seed)
    if xgb is not None:
        makers["XGBoost"] = lambda: xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", verbosity=0, random_state=c.cfg.seed)
    lines = [
        "Ensembles of decision trees fit to P(open->close > 0 | features); RF = bagging + random feature subsets; boosting = additive trees fit to gradient of log-loss",
        f"{X.shape[1]} features x {n} days, walk-forward OOS ({tscv.n_splits} folds), base rate up={up.mean()*100:.1f}%",
    ]
    sigs, ps, aucs = [], {}, []
    for name, mk in makers.items():
        pr = np.full(n, np.nan)
        for tr, te in tscv.split(X):
            m = mk().fit(X.iloc[tr], up.iloc[tr])
            pr[te] = m.predict_proba(X.iloc[te])[:, 1]
        ok = ~np.isnan(pr)
        acc = float(((pr[ok] > 0.5) == (up.values[ok] == 1)).mean())
        try:
            auc = float(roc_auc_score(up.values[ok], pr[ok]))
        except Exception:
            auc = 0.5
        m = mk().fit(X, up)
        p = float(m.predict_proba(live)[:, 1][0])
        q = _skill(acc, 0.0, auc)
        sigs.append(q * 2 * (p - 0.5) * 3)
        ps[name] = p
        aucs.append(auc)
        imp = getattr(m, "feature_importances_", None)
        top = ""
        if imp is not None:
            s = pd.Series(imp, index=X.columns).sort_values(ascending=False)[:4]
            top = "  top: " + ", ".join(f"{k} {v:.2f}" for k, v in s.items())
        lines.append(f"{name:<12} OOS acc={acc*100:.1f}%  AUC={auc:.3f}  skill-weight={q:.2f}  P(up today)={p*100:.1f}%{top}")
    c.info.update(tree_p=float(np.mean(list(ps.values()))), tree_auc=float(np.mean(aucs)), tree_models="/".join(ps.keys()))
    R.signal = clip(float(np.mean(sigs)))
    R.headline = " | ".join(f"{k} P(up)={v*100:.0f}%" for k, v in ps.items())
    R.lines = lines + ["Votes are scaled by out-of-sample skill; a model at chance (AUC~0.5) contributes ~0."]
    return R


def _sim_exec(policy, Q, n_ep, T, nI, sig, eta, g, lam, rng):
    """Implementation shortfall in bps per unit for a policy on simulated price paths."""
    costs = np.empty(n_ep)
    Z = rng.standard_normal((n_ep, T))
    for e in range(n_ep):
        P, inv, sold, tot = 0.0, nI, 0, 0.0
        for t in range(T):
            P += sig * Z[e, t]
            a = inv if t == T - 1 else min(policy(t, inv, Q), inv)
            tot += a * (P - eta * a - g * sold)
            sold += a
            inv -= a
        costs[e] = -tot / nI
    return costs


def mod_rl_exec(c: Ctx) -> Result:
    R = Result("rl", "RL execution agent (Q-learning / PPO)", "4. Machine learning & execution", weight=0.0)
    rng = np.random.default_rng(c.cfg.seed)
    df = c.intra
    bar_bps = float(np.std(np.log(df["close"] / df["open"]).values)) * 1e4 if df is not None and len(df) > 200 else 8.0
    T, nI, A = 10, 10, 5                                   # 10 steps of 6 min, 10 units, up to 4 units/step
    sig = bar_bps * math.sqrt(6 / 5)
    eta, g, lam = 0.04, 0.01, 0.002
    Q = np.zeros((T, nI + 1, A))
    alpha_lr = 0.08
    N_EP = 40_000
    for ep in range(N_EP):
        P, inv, sold = 0.0, nI, 0
        eps = max(0.05, 1.0 - ep / (0.6 * N_EP))
        for t in range(T):
            P += sig * rng.standard_normal()
            amax = min(A - 1, inv)
            if t == T - 1:
                a_exec, a_idx = inv, min(inv, A - 1)
            else:
                a_idx = int(rng.integers(0, amax + 1)) if rng.random() < eps else int(np.argmax(Q[t, inv, :amax + 1]))
                a_exec = a_idx
            rew = a_exec * (P - eta * a_exec - g * sold)
            inv2 = inv - a_exec
            rew -= lam * (inv2 * sig) ** 2
            tgt = rew if t == T - 1 else rew + np.max(Q[t + 1, inv2, :min(A - 1, inv2) + 1])
            Q[t, inv, a_idx] += alpha_lr * (tgt - Q[t, inv, a_idx])
            sold += a_exec
            inv = inv2
    greedy = lambda t, inv, Q_: int(np.argmax(Q_[t, inv, :min(A - 1, inv) + 1]))
    twap = lambda t, inv, Q_: 1
    front = lambda t, inv, Q_: min(4, inv)
    n_eval = 8000
    seed = int(rng.integers(1e9))
    cq = _sim_exec(greedy, Q, n_eval, T, nI, sig, eta, g, lam, np.random.default_rng(seed))
    ct = _sim_exec(twap, Q, n_eval, T, nI, sig, eta, g, lam, np.random.default_rng(seed))
    cf = _sim_exec(front, Q, n_eval, T, nI, sig, eta, g, lam, np.random.default_rng(seed))
    inv, sched = nI, []
    for t in range(T):
        a = inv if t == T - 1 else greedy(t, inv, Q)
        sched.append(a)
        inv -= a
    lines = [
        "MDP: state (t, inventory), action a = units sold this 6-min step, reward = a (P_t - eta a - g sold_so_far) - lambda (inventory x sigma)^2  (Almgren-Chriss-style impact + risk)",
        f"Q(s,a) <- Q + alpha [ r + max_a' Q(s',a') - Q ],  epsilon-greedy (1.0 -> 0.05),  {N_EP:,} episodes;  sigma_step={sig:.1f}bp  eta={eta}bp/unit^2  g={g}  lambda={lam}",
        f"Learned schedule (units per 6 min): {sched}",
        f"Shortfall bps/unit (lower=better; mean +/- sd): Q-learning {cq.mean():.2f} +/- {cq.std():.1f} | TWAP {ct.mean():.2f} +/- {ct.std():.1f} | front-loaded {cf.mean():.2f} +/- {cf.std():.1f}",
    ]
    c.info.update(rl_sched=sched, rl_cost=float(cq.mean()), rl_twap=float(ct.mean()), rl_front=float(cf.mean()))
    ppo_txt = ""
    if c.cfg.ppo:
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO

            class ExecEnv(gym.Env):
                def __init__(s):
                    s.observation_space = gym.spaces.Box(0.0, 1.0, (2,), np.float32)
                    s.action_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)

                def reset(s, seed=None, options=None):
                    s.t, s.inv, s.sold, s.P = 0, nI, 0.0, 0.0
                    return np.array([0, 1], np.float32), {}

                def step(s, act):
                    s.P += sig * np.random.randn()
                    a = s.inv if s.t == T - 1 else float(act[0]) * s.inv
                    rew = a * (s.P - eta * a - g * s.sold) - lam * ((s.inv - a) * sig) ** 2
                    s.sold += a
                    s.inv -= a
                    s.t += 1
                    return np.array([s.t / T, s.inv / nI], np.float32), float(rew) / 10.0, s.t >= T, False, {}
            model = PPO("MlpPolicy", ExecEnv(), verbose=0, n_steps=512, batch_size=128, seed=c.cfg.seed).learn(60_000)
            tot = []
            for _ in range(1500):
                env = ExecEnv(); ob, _ = env.reset(); done, ret_ = False, 0.0
                while not done:
                    ob, rw, done, _, _ = env.step(model.predict(ob, deterministic=True)[0])
                    ret_ += rw * 10.0
                tot.append(-ret_ / nI)
            ppo_txt = f" | PPO {np.mean(tot):.2f}"
            lines.append(f"PPO (stable-baselines3, continuous action = fraction of remaining inventory, 60k steps): shortfall {np.mean(tot):.2f} +/- {np.std(tot):.1f} bps/unit")
        except Exception as e:
            lines.append(f"PPO skipped: {e}")
    else:
        lines.append("PPO: off (use --ppo; needs gymnasium + stable-baselines3).")
    lines.append("Execution models are not directional; they estimate how much slippage to expect and how to slice a parent order.")
    R.headline = f"schedule {sched} | shortfall bps/unit: Q-learn {cq.mean():.2f} vs TWAP {ct.mean():.2f} vs front {cf.mean():.2f}{ppo_txt}"
    R.lines = lines
    return R


# =============================================================================
# ORCHESTRATION: run every model, fuse into one composite, build the report
# =============================================================================
MODULES: List[Tuple[Callable, str]] = [
    (mod_gbm_ito, "gbm"), (mod_bsm, "bsm"), (mod_heston, "heston"), (mod_sabr, "sabr"), (mod_rates, "rates"),
    (mod_arima, "arima"), (mod_garch, "garch"), (mod_pairs, "pairs"), (mod_factors, "factors"), (mod_markowitz, "markowitz"), (mod_hmm, "hmm"),
    (mod_montecarlo, "mc"), (mod_fdm, "fdm"), (mod_rootfind, "root"), (mod_lattice, "lattice"),
    (mod_hawkes, "hawkes"), (mod_avellaneda, "as"), (mod_linear_ml, "linear"), (mod_trees, "trees"), (mod_rl_exec, "rl"),
]
# volatility models first: later modules (Monte Carlo, Avellaneda-Stoikov, RL) reuse their sigma
RUN_ORDER = ["gbm", "garch", "bsm", "heston", "sabr", "rates", "arima", "pairs", "factors", "markowitz", "hmm",
             "mc", "fdm", "root", "lattice", "hawkes", "as", "linear", "trees", "rl"]
TITLES = {
    "gbm": ("GBM & Ito's lemma", "1. Derivative pricing & stochastic calculus"), "bsm": ("Black-Scholes & Greeks", "1. Derivative pricing & stochastic calculus"),
    "heston": ("Heston", "1. Derivative pricing & stochastic calculus"), "sabr": ("SABR", "1. Derivative pricing & stochastic calculus"),
    "rates": ("Vasicek / CIR / Hull-White", "1. Derivative pricing & stochastic calculus"),
    "arima": ("ARIMA / SARIMAX", "2. Statistical & time-series"), "garch": ("GARCH", "2. Statistical & time-series"),
    "pairs": ("Cointegration + OU", "2. Statistical & time-series"), "factors": ("CAPM & Fama-French", "2. Statistical & time-series"),
    "markowitz": ("Markowitz", "2. Statistical & time-series"), "hmm": ("Hidden Markov regimes", "2. Statistical & time-series"),
    "mc": ("Monte Carlo + variance reduction", "3. Numerical & computational"), "fdm": ("Finite differences", "3. Numerical & computational"),
    "root": ("Newton / Brent root-finding", "3. Numerical & computational"), "lattice": ("Binomial & trinomial trees", "3. Numerical & computational"),
    "hawkes": ("Hawkes process", "4. Machine learning & execution"), "as": ("Avellaneda-Stoikov", "4. Machine learning & execution"),
    "linear": ("Lasso / Ridge / ElasticNet", "4. Machine learning & execution"), "trees": ("RF / LightGBM / XGBoost", "4. Machine learning & execution"),
    "rl": ("RL execution (Q-learning / PPO)", "4. Machine learning & execution"),
}


def run_all(c: Ctx) -> List[Result]:
    fns = {k: fn for fn, k in MODULES}
    out: Dict[str, Result] = {}
    for key in RUN_ORDER:
        t0 = time.time()
        title, sec = TITLES[key]
        try:
            r = fns[key](c)
            r.lines.append(f"(computed in {time.time()-t0:.1f}s)")
            log.info("%-10s signal %+.2f  %s", key, r.signal, r.headline[:90])
        except Exception as e:
            log.warning("%s failed: %s", key, e)
            log.debug(traceback.format_exc())
            r = Result(key, title, sec, ok=False, headline=f"skipped - {type(e).__name__}: {e}"[:200])
        if r.ok and key in WEIGHT_OVERRIDE:
            r.weight = WEIGHT_OVERRIDE[key]
        out[key] = r
    return [out[k] for _, k in MODULES]


@dataclass
class Composite:
    score: float
    p_up: float
    verdict: str
    agree: float
    move: float
    move_parts: Dict[str, float]
    center: float
    lv: List[Tuple[str, float, str]]        # (name, price, why it matters)


def compose(res: List[Result], c: Ctx) -> Composite:
    live = [r for r in res if r.ok and r.weight > 0]
    W = sum(r.weight for r in live) or 1.0
    score = sum(r.weight * r.signal for r in live) / W
    signed = [r for r in live if abs(r.signal) > 0.05]
    agree = (sum(1 for r in signed if np.sign(r.signal) == np.sign(score)) / len(signed)) if signed else 0.0
    p_up = 0.5 + 0.25 * clip(score)                                    # deliberately conservative mapping
    thr = c.cfg.bias_threshold
    verdict = "DOWN" if score < -thr else "UP" if score > thr else "NEUTRAL / NO EDGE"
    parts = {}
    if "implied_move" in c.sig:
        parts["options-implied (straddle)"] = c.sig["implied_move"]
    for k, lab in (("garch", "GARCH"), ("heston", "Heston MC"), ("gbm", "realised (252d)")):
        if k in c.sig:
            parts[lab] = c.sig[k]
    wts = {"options-implied (straddle)": 0.4, "GARCH": 0.3, "Heston MC": 0.15, "realised (252d)": 0.15}
    tw = sum(wts[k] for k in parts) or 1.0
    move = sum(wts[k] * v for k, v in parts.items()) / tw if parts else 0.012
    S = c.spot
    center = S * math.exp(score * 0.15 * move)
    p1 = float(Ncdf(-1.0)) * 100                     # ~16% of days close beyond 1 sigma (each side)
    p2 = float(Ncdf(-2.0)) * 100                     # ~2% beyond 2 sigma
    lv = [
        ("Stretched upside", S * math.exp(2 * move), f"a 2-sigma rally; only ~{p2:.0f}% of days close above, so it's an exhaustion zone, not a target"),
        ("Implied upside base", S * math.exp(move), f"a normal good day; ~{p1:.0f}% of days close above. Today's realistic ceiling"),
        ("Model centre", center, "where the weighted model bias puts the close; it barely moves off the open by design"),
        ("Implied downside base", S * math.exp(-move), f"a normal bad day; ~{p1:.0f}% of days close below. Today's realistic floor"),
        ("Stretched downside", S * math.exp(-2 * move), f"a 2-sigma drop; only ~{p2:.0f}% of days close below, so it's a capitulation zone"),
    ]
    return Composite(score, p_up, verdict, agree, move, parts, center, lv)


# =============================================================================
# DESK CONTEXT -- confluence filters, read together the way a trading desk reads the tape
# =============================================================================
EXTRA_TICKERS = ["^VXN", "^VIX9D", "^VIX3M", "HYG", "LQD", "DX-Y.NYB", "SMH"]

# Vote weights re-set from the full walk-forward backtest (backtest_full.py, Jan-2025 -> Oct-2026):
# tree models were the only clearly-above-chance vote (56.8%); factors / Markowitz / Avellaneda-Stoikov
# were below 50%; ARIMA, GBM and Heston drift were coin flips or always-long.
WEIGHT_OVERRIDE = {"trees": 2.0, "hmm": 0.5, "garch": 0.5, "arima": 0.2, "heston": 0.2, "gbm": 0.1,
                   "factors": 0.0, "markowitz": 0.0, "as": 0.0}


def _last(px, t, back=0):
    if t not in px:
        return float("nan")
    s_ = px[t].dropna()
    return float(s_.iloc[-1 - back]) if len(s_) > back else float("nan")


def _lchg(px, t, n):
    a, b = _last(px, t), _last(px, t, n)
    return math.log(a / b) if a == a and b == b and a > 0 and b > 0 else float("nan")


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def dealer_gamma(c: Ctx) -> Optional[dict]:
    """Dealer gamma exposure from the live chain (SqueezeMetrics convention: customers buy calls and puts,
    so dealers are long call gamma and short put gamma). GEX = sum gamma x OI x 100 x S^2 x 1%  (in $ per 1% move).
    Also: gamma flip (spot where total GEX changes sign), call wall, put wall, max pain, put/call ratios."""
    S, r, q = c.spot, c.rf, c.q
    K_, sgn, oi_, iv_, T_, vol_ = [], [], [], [], [], []
    near_rows = []
    for name in ("near", "term"):
        ch = getattr(c, name)
        if ch is None:
            continue
        df, T = ch["df"], ch["T"]
        for _, o in df.iterrows():
            oi = float(o.get("openInterest", 0) or 0)
            K = float(o["strike"])
            if oi <= 0 or not (0.85 * S < K < 1.15 * S):
                continue
            iv = float(o.get("impliedVolatility", 0) or 0)
            if not (0.03 < iv < 3.0):
                m = _mid(o)
                iv = iv_brent(m, S, K, T, r, q, o["kind"])[0] if m > 0.02 else float("nan")
            if not _ok(iv):
                continue
            K_.append(K); sgn.append(1.0 if o["kind"] == "c" else -1.0); oi_.append(oi); iv_.append(iv); T_.append(T)
            vol_.append(float(o.get("volume", 0) or 0))
            if name == "near":
                near_rows.append((K, o["kind"], oi))
    if len(K_) < 12:
        return None
    K_, sgn, oi_, iv_, T_, vol_ = map(np.asarray, (K_, sgn, oi_, iv_, T_, vol_))

    def gamma_at(Sx):
        sT = iv_ * np.sqrt(T_)
        d1 = (np.log(Sx / K_) + (r - q + 0.5 * iv_ ** 2) * T_) / sT
        return np.exp(-q * T_) * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (Sx * sT)

    def gex(Sx):
        return float(np.sum(sgn * gamma_at(Sx) * oi_ * 100 * Sx * Sx * 0.01))
    g0 = gex(S)
    grid = np.linspace(0.92 * S, 1.08 * S, 161)
    vals = np.array([gex(x) for x in grid])
    flips = [grid[i] - vals[i] * (grid[i + 1] - grid[i]) / (vals[i + 1] - vals[i])
             for i in range(len(grid) - 1) if np.sign(vals[i]) != np.sign(vals[i + 1])]
    flip = min(flips, key=lambda x: abs(x - S)) if flips else None
    per = pd.DataFrame({"K": K_, "g": gamma_at(S) * oi_, "s": sgn})
    calls = per[(per.s > 0) & (per.K > S)].groupby("K").g.sum()
    puts = per[(per.s < 0) & (per.K < S)].groupby("K").g.sum()
    call_wall = float(calls.idxmax()) if len(calls) else None
    put_wall = float(puts.idxmax()) if len(puts) else None
    max_pain = None
    if near_rows:
        nr = pd.DataFrame(near_rows, columns=["K", "kind", "oi"])
        strikes = np.sort(nr.K.unique())
        pay = [float((nr[nr.kind == "c"].oi * np.maximum(k - nr[nr.kind == "c"].K, 0)).sum()
                     + (nr[nr.kind == "p"].oi * np.maximum(nr[nr.kind == "p"].K - k, 0)).sum()) for k in strikes]
        max_pain = float(strikes[int(np.argmin(pay))])
    c_oi, p_oi = oi_[sgn > 0].sum(), oi_[sgn < 0].sum()
    c_v, p_v = vol_[sgn > 0].sum(), vol_[sgn < 0].sum()
    return dict(gex=g0, flip=flip, call_wall=call_wall, put_wall=put_wall, max_pain=max_pain,
                pc_oi=p_oi / c_oi if c_oi > 0 else float("nan"), pc_vol=p_v / c_v if c_v > 0 else float("nan"))


def _third_friday(y, m):
    from datetime import date as D
    d = D(y, m, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(days=14)


def desk_context(c: Ctx, k: Composite) -> dict:
    """Every confluence filter, then the regimes they imply. Cached in c.info['desk']."""
    if "desk" in c.info:
        return c.info["desk"]
    S, px, o, m = c.spot, c.px, c.ohlc, max(k.move, 1e-6)
    cl, hi, lo = o["close"], o["high"], o["low"]
    d: Dict[str, object] = {}
    # --- trend & stretch ---
    for n in (20, 50, 200):
        d[f"sma{n}"] = float(cl.iloc[-n:].mean()) if len(cl) >= n else float("nan")
    d["mom21"] = math.log(cl.iloc[-1] / cl.iloc[-22]) if len(cl) > 22 else float("nan")
    d["mom63"] = math.log(cl.iloc[-1] / cl.iloc[-64]) if len(cl) > 64 else float("nan")
    sd20 = float(cl.iloc[-20:].std())
    d["z20"] = (S - d["sma20"]) / sd20 if sd20 > 0 else float("nan")
    d["rsi2"] = float(_rsi(cl, 2).iloc[-1])
    # --- yesterday's structure ---
    d["pdh"], d["pdl"], d["pdc"] = float(hi.iloc[-1]), float(lo.iloc[-1]), float(cl.iloc[-1])
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = float(tr.iloc[-14:].mean())
    rng = (hi - lo).iloc[-7:]
    d["nr7"] = bool(len(rng) == 7 and rng.iloc[-1] <= rng.min() + 1e-12)
    d["prior_vwap"] = d["prior_poc"] = float("nan")
    if c.intra is not None and len(c.intra):
        it = c.intra
        dates = (it.index.tz_convert(ET) if it.index.tz is not None else it.index).date
        last = it[dates == dates[-1]]
        if len(last) > 10 and "volume" in last:
            tp = (last["high"] + last["low"] + last["close"]) / 3
            v = last["volume"].astype(float)
            if v.sum() > 0:
                d["prior_vwap"] = float((tp * v).sum() / v.sum())
                b = (tp / 0.25).round() * 0.25
                d["prior_poc"] = float(v.groupby(b).sum().idxmax())
    d["pm_high"] = d["pm_low"] = float("nan")
    if c.premkt is not None and len(c.premkt) > 5:
        d["pm_high"], d["pm_low"] = float(c.premkt["high"].max()), float(c.premkt["low"].min())
    # --- overnight gap ---
    d["gap"] = S / c.prev_close - 1
    d["gap_z"] = d["gap"] / m
    # --- volatility complex ---
    vix, v9, v3, vxn = (_last(px, t) for t in ("^VIX", "^VIX9D", "^VIX3M", "^VXN"))
    d.update(vix=vix, vix9d=v9, vix3m=v3, vxn=vxn,
             ts_front=v9 / vix if _ok(v9) and _ok(vix) and vix > 0 else float("nan"),
             ts_back=vix / v3 if _ok(vix) and _ok(v3) and v3 > 0 else float("nan"))
    # --- cross-asset ---
    d["credit5"] = _lchg(px, "HYG", 5) - _lchg(px, "LQD", 5)
    d["dxy5"] = _lchg(px, "DX-Y.NYB", 5)
    t10a, t10b = _last(px, "^TNX"), _last(px, "^TNX", 5)
    d["y10_5d_bp"] = (t10a - t10b) * 100 if _ok(t10a) and _ok(t10b) else float("nan")
    d["semis5"] = _lchg(px, "SMH", 5) - _lchg(px, c.cfg.equity, 5)
    d["vs_spy5"] = _lchg(px, c.cfg.equity, 5) - _lchg(px, c.cfg.pair, 5)
    # --- options positioning ---
    g = None
    try:
        g = dealer_gamma(c)
    except Exception as e:
        log.warning("dealer gamma failed: %s", e)
    d["gamma"] = g
    # --- calendar ---
    today = c.now.date()
    tf = _third_friday(today.year, today.month)
    d["opex_week"] = 0 <= (tf - today).days <= 4
    d["opex_day"] = today == tf
    month_days = [x for x in pd.bdate_range(today.replace(day=1), periods=23) if x.month == today.month]
    pos = [x.date() for x in month_days].index(today) if today in [x.date() for x in month_days] else 5
    d["turn_of_month"] = pos <= 2 or pos >= len(month_days) - 1

    # ================= regimes =================
    i = c.info
    stress = 0
    stress += 2 if _ok(d["ts_back"]) and d["ts_back"] > 1.0 else 0
    stress += 1 if _ok(d["ts_front"]) and d["ts_front"] > 1.05 else 0
    stress += 1 if i.get("garch_pctl", 0.5) > 0.80 else 0
    stress += 1 if str(i.get("hmm_state", "")).startswith("STRESS") and i.get("hmm_prob", 0) > 0.6 else 0
    stress += 1 if _ok(vxn) and vxn > 30 else 0
    calm = (_ok(d["ts_back"]) and d["ts_back"] < 0.92) + (i.get("garch_pctl", 0.5) < 0.35)
    d["vol_regime"] = "stressed" if stress >= 3 else "elevated" if stress >= 1 else "calm" if calm >= 1 else "normal"

    d["gamma_regime"] = "unknown"
    if g:
        if g["gex"] > 0 and (g["flip"] is None or S > g["flip"]):
            d["gamma_regime"] = "long"
        elif g["gex"] < 0 or (g["flip"] is not None and S < g["flip"]):
            d["gamma_regime"] = "short"

    s50, s200 = d["sma50"], d["sma200"]
    if _ok(s200) and S > s50 > s200 and d["mom63"] > 0:
        d["trend"] = "uptrend"
    elif _ok(s200) and S < s50 < s200 and d["mom63"] < 0:
        d["trend"] = "downtrend"
    else:
        d["trend"] = "range"

    ra = 0.0
    for val, w in ((d["credit5"], 1.0), (-d["dxy5"] if _ok(d["dxy5"]) else float("nan"), 0.5),
                   (-d["y10_5d_bp"] if _ok(d["y10_5d_bp"]) else float("nan"), 0.5), (d["semis5"], 1.0)):
        if _ok(val) and abs(val) > 1e-9:
            ra += w * math.copysign(1, val)
    d["risk_score"] = ra
    d["risk"] = "risk-on" if ra >= 1.5 else "risk-off" if ra <= -1.5 else "mixed"

    if (_ok(d["rsi2"]) and d["rsi2"] > 90) or (_ok(d["z20"]) and d["z20"] > 2):
        d["stretch"] = "stretched up"
    elif (_ok(d["rsi2"]) and d["rsi2"] < 10) or (_ok(d["z20"]) and d["z20"] < -2):
        d["stretch"] = "stretched down"
    else:
        d["stretch"] = "not stretched"

    why = []
    if d["vol_regime"] == "stressed":
        why.append("volatility is stressed")
    if d["gamma_regime"] == "short":
        why.append("dealers are short gamma")
    if d["nr7"]:
        why.append("yesterday was the narrowest range in 7 days (NR7, ranges tend to expand after)")
    if abs(d["gap_z"]) > 0.75:
        why.append(f"the opening gap is already {abs(d['gap_z'])*100:.0f}% of a normal day")
    if why:
        d["day_type"], d["stance"] = "EXPANSION / TREND DAY", "trade breaks with confirmation, don't fade, smaller size"
    elif d["gamma_regime"] == "long" and d["vol_regime"] in ("calm", "normal"):
        d["day_type"], d["stance"] = "MEAN-REVERSION / RANGE DAY", "buy confirmed dips into support, take profits into resistance"
        why = ["dealers are long gamma (they sell rips and buy dips, dampening moves)", f"volatility is {d['vol_regime']}"]
    else:
        d["day_type"], d["stance"] = "BALANCED DAY", "wait for price to reach a confluence zone before acting"
        why = [f"volatility is {d['vol_regime']}"] + (["dealer gamma is unavailable"] if d["gamma_regime"] == "unknown" else [])
    d["day_why"] = why

    # directional context agreement with the bias
    side = 1 if k.verdict == "UP" else -1 if k.verdict == "DOWN" else 0
    votes = {"trend": {"uptrend": 1, "downtrend": -1}.get(d["trend"], 0),
             "cross-asset": {"risk-on": 1, "risk-off": -1}.get(d["risk"], 0),
             "stretch (mean reversion)": {"stretched up": -1, "stretched down": 1}.get(d["stretch"], 0)}
    if g and _ok(g.get("max_pain")):
        votes["options pin (max pain)"] = 1 if g["max_pain"] > S * (1 + 0.15 * m) else -1 if g["max_pain"] < S * (1 - 0.15 * m) else 0
    d["ctx_votes"] = votes
    lean = sum(votes.values())
    d["ctx_lean"] = lean
    if side:
        agree = sum(1 for v in votes.values() if v == side)
        against = sum(1 for v in votes.values() if v == -side)
        d["conviction"] = "high" if agree >= 2 and against == 0 else "low" if against >= agree else "medium"
        d["agree"], d["against"] = agree, against
    else:
        d["conviction"] = "n/a"
        d["agree"] = d["against"] = 0
    c.info["desk"] = d
    return d


def confluence_zones(c: Ctx, k: Composite) -> List[dict]:
    """Every level from every method, clustered into zones; a zone is stronger when independent methods agree."""
    S, m, d = c.spot, max(k.move, 1e-6), desk_context(c, k)
    src = []
    for v, nm, what, react in level_ladder(c, k):
        if nm in ("OPEN", "ML CLOSE TARGET"):
            continue
        fam = "order flow (Hawkes)" if "MOMENTUM" in nm else "relative value" if "FAIR" in nm else "volatility model"
        src.append(dict(price=v, name=nm, family=fam, react=react, what=what))
    add = [("pdh", "PRIOR-DAY HIGH", "yesterday's structure"), ("pdl", "PRIOR-DAY LOW", "yesterday's structure"),
           ("pdc", "PRIOR CLOSE", "yesterday's structure"), ("prior_vwap", "PRIOR-SESSION VWAP", "yesterday's volume"),
           ("prior_poc", "PRIOR-SESSION POC", "yesterday's volume"), ("pm_high", "PREMARKET HIGH", "overnight structure"),
           ("pm_low", "PREMARKET LOW", "overnight structure"), ("sma20", "20-DAY AVERAGE", "trend averages"),
           ("sma50", "50-DAY AVERAGE", "trend averages"), ("sma200", "200-DAY AVERAGE", "trend averages")]
    for key, nm, fam in add:
        v = d.get(key)
        if _ok(v) and abs(math.log(v / S)) < 2.5 * m:
            src.append(dict(price=float(v), name=nm, family=fam, react="", what=""))
    g = d.get("gamma")
    if g:
        for key, nm in (("flip", "GAMMA FLIP"), ("call_wall", "CALL WALL"), ("put_wall", "PUT WALL"), ("max_pain", "MAX PAIN")):
            v = g.get(key)
            if _ok(v) and abs(math.log(v / S)) < 2.5 * m:
                src.append(dict(price=float(v), name=nm, family="options positioning", react="", what=""))
    src.sort(key=lambda x: x["price"])
    zones, tol = [], 0.15 * m * S
    for x in src:
        if zones and x["price"] - zones[-1]["hi"] <= tol:
            zones[-1]["items"].append(x)
            zones[-1]["hi"] = x["price"]
        else:
            zones.append(dict(lo=x["price"], hi=x["price"], items=[x]))
    for z in zones:
        z["price"] = float(np.mean([x["price"] for x in z["items"]]))
        z["families"] = sorted(set(x["family"] for x in z["items"]))
        z["names"] = [x["name"] for x in z["items"]]
        main = next((x for x in z["items"] if x["family"] == "volatility model"), z["items"][0])
        z["main"] = main
    return zones


def fut(c: Ctx, x: float) -> str:
    return f"{x*c.fut_ratio:,.0f}" if c.fut_ratio == c.fut_ratio else "n/a"


def _one_in(p: float) -> str:
    return f"about 1 in {max(2, round(1 / p))}" if p and p > 0 else "very unlikely"


def _ord(n: float) -> str:
    n = int(round(n))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def bias_text(c: Ctx, k: Composite, res: List[Result]) -> str:
    """Plain-English directional bias with the odds behind it."""
    live = [r for r in res if r.ok and r.weight > 0]
    side = 1 if k.score >= 0 else -1
    drivers = sorted([r for r in live if r.signal * side > 0.05], key=lambda r: -abs(r.weight * r.signal))[:2]
    drv = " and ".join(r.title.split(" (")[0] for r in drivers) or "no single model"
    strength = "weak" if abs(k.score) < 0.25 else "moderate" if abs(k.score) < 0.45 else "strong"
    lv = {nm: v for nm, v, _ in k.lv}
    if k.verdict == "UP":
        head = f"LEAN UP ({strength}). Most models tilt bullish"
    elif k.verdict == "DOWN":
        head = f"LEAN DOWN ({strength}). Most models tilt bearish"
    else:
        head = "NO CLEAR BIAS. The models are split"
    t = (f"{head} (score {k.score:+.2f} on a -1 to +1 scale, {k.agree*100:.0f}% of models agree, strongest pull from {drv}). "
         f"Rough odds QQQ closes above the open: {k.p_up*100:.0f}%. ")
    if k.verdict == "NEUTRAL / NO EDGE":
        t += (f"Treat today as a range day: a normal session stays between {f(lv['Implied downside base'])} and "
              f"{f(lv['Implied upside base'])} (+/-{pc(k.move,2)}). ")
    else:
        t += (f"That is an edge, not a certainty. A normal session still ranges {f(lv['Implied downside base'])} to "
              f"{f(lv['Implied upside base'])} (+/-{pc(k.move,2)}). ")
    return t + "The range is more reliable than the direction."


def donts_list(c: Ctx, k: Composite) -> List[str]:
    S, info = c.spot, c.info
    lv = {nm: v for nm, v, _ in k.lv}
    donts: List[str] = []
    fair, z = info.get("pair_fair"), info.get("pair_z")
    if fair is not None and z is not None:
        loose = "" if info.get("pair_coint") else " (the SPY link is weak right now, so treat this loosely)"
        if z > 1.0:
            donts.append(f"Don't buy right now: QQQ ({f(S)}) is {pc(S/fair-1,1)} above its fair value vs SPY ({f(fair)}). "
                         f"Stretched prices tend to pull back, so buying here means paying a premium{loose}.")
        elif z < -1.0:
            donts.append(f"Don't short right now: QQQ ({f(S)}) is {pc(1-S/fair,1)} below its fair value vs SPY ({f(fair)}). "
                         f"Cheap prices tend to drift back up{loose}.")
    vrp, iv, rv = info.get("vrp"), info.get("iv"), info.get("rv21")
    if vrp is not None:
        if vrp > 0.02:
            donts.append(f"Don't buy options today: Black-Scholes shows they're priced for {pc(iv,0)} yearly movement, "
                         f"but QQQ has only been moving {pc(rv,0)}. You'd be overpaying.")
        elif vrp < -0.02:
            donts.append(f"Don't sell options today: they're priced for {pc(iv,0)} yearly movement while QQQ is actually moving "
                         f"{pc(rv,0)}, so they're cheap and selling them is risky.")
    gap = S / c.prev_close - 1
    if abs(gap) > 0.75 * k.move:
        donts.append(f"Don't chase the opening gap: QQQ already moved {pc(gap,2)} overnight, "
                     f"{abs(gap)/k.move*100:.0f}% of a normal day's move.")
    pctl = info.get("garch_pctl")
    if pctl is not None and pctl > 0.80:
        donts.append(f"Don't trade full size: volatility is busier than {pctl*100:.0f}% of past days (GARCH), so stops get hit more easily.")
    if str(info.get("hmm_state", "")).startswith("STRESS") and info.get("hmm_prob", 0) > 0.6:
        donts.append("Don't assume dips get bought: the market-state model says we're in a high-volatility phase, where moves run further than usual.")
    donts.append("Don't short into the upside limit or extreme high: in the 2025-26 backtest that trade lost 2,069 NQ points (39% win rate).")
    dk = desk_context(c, k)
    if dk.get("conviction") == "low":
        donts.append(f"Don't size up on the {k.verdict.lower()} bias: the context ({', '.join(n for n, v in dk['ctx_votes'].items() if v and v != (1 if k.verdict == 'UP' else -1))}) disagrees with it.")
    if k.verdict == "UP":
        donts.append(f"Don't short just because it looks high: below {f(lv['Implied upside base'])} the bias favours buyers.")
    elif k.verdict == "DOWN":
        donts.append(f"Don't buy dips just because it looks low: above {f(lv['Implied downside base'])} the bias favours sellers.")
    else:
        donts.append(f"Don't force a trade in the middle of the range: wait for {f(lv['Implied downside base'])} or {f(lv['Implied upside base'])}.")
    return donts


def level_ladder(c: Ctx, k: Composite) -> List[Tuple[float, str, str, str]]:
    """(price, name, what it is, how to react) from highest to lowest."""
    S, i, b = c.spot, c.info, k.verdict
    lv = {nm: v for nm, v, _ in k.lv}
    up, dn = b == "UP", b == "DOWN"
    p1 = _one_in(float(Ncdf(-1.0)))
    p2 = _one_in(float(Ncdf(-2.0)))
    L = [
        (lv["Stretched upside"], "EXTREME HIGH", f"A 2-sigma rally (twice a normal day). {p2.capitalize()} days close above this.",
         "Take most profits on longs; don't chase higher." if up else
         "If price gets here, the down bias is wrong. Get out of shorts." if dn else
         "Very stretched. Take profits on longs here; don't open shorts."),
        (lv["Implied upside base"], "UPSIDE LIMIT", f"The top of a normal day (1-sigma). {p1.capitalize()} days close above this.",
         f"First profit target for longs. A strong break above opens the way to {f(lv['Stretched upside'])}." if up else
         "Rallies tend to stall here on down-bias days. Don't buy here; don't short it either (shorting this level lost money in testing)." if dn else
         "Top of today's likely range. Take profits on longs here; don't open shorts (shorting it lost money in testing)."),
        (S, "OPEN", "Today's opening price, the reference point for the bias.",
         "Holding above here keeps the up bias alive; losing it early is a warning sign." if up else
         "Staying below here keeps the down bias alive; reclaiming it is a warning sign." if dn else
         "Expect chop around here. Patience: no edge in the middle."),
        (lv["Implied downside base"], "DOWNSIDE LIMIT", f"The bottom of a normal day (1-sigma). {p1.capitalize()} days close below this.",
         "Best area to look for longs: wait for price to stop falling (a 5-min candle closing back above)." if up else
         f"First profit target for shorts. A strong break below opens the way to {f(lv['Stretched downside'])}." if dn else
         "Bottom of today's likely range. Buy dips here rather than sell them."),
        (lv["Stretched downside"], "EXTREME LOW", f"A 2-sigma drop (twice a normal day). {p2.capitalize()} days close below this.",
         "If price gets here, the up bias is wrong. Get out of longs." if up else
         "Take most profits on shorts; don't chase lower." if dn else
         "Very stretched. Only consider a small long if price stalls here."),
    ]
    if "hawkes_rho" in i and i["hawkes_rho"] < 0.97:
        burst = i["hawkes_thr"] / (1 - i["hawkes_rho"])
        L.append((S * math.exp(burst), "MOMENTUM TRIGGER UP", "Where a chain reaction of buying would typically carry price (Hawkes model).",
                  "If price rushes through here fast, it's a buying burst. Don't short into it; let it slow down first."))
        L.append((S * math.exp(-burst), "MOMENTUM TRIGGER DOWN", "Where a chain reaction of selling would typically carry price (Hawkes model).",
                  "If price crashes through here fast, it's a selling burst. Don't try to catch it; let it slow down first."))
    targets = []
    if "reg_pred" in i:
        targets.append(S * math.exp(i["reg_pred"] / 100))
    if "tree_p" in i:
        targets.append(S * math.exp((2 * i["tree_p"] - 1) * k.move * math.sqrt(2 / math.pi)))
    if targets:
        t = float(np.mean(targets))
        detail = []
        if "reg_pred" in i:
            detail.append(f"regression {i['reg_pred']:+.2f}%")
        if "tree_p" in i:
            detail.append(f"tree models {i['tree_p']*100:.0f}% chance of up")
        L.append((t, "ML CLOSE TARGET", "Where the machine-learning models expect the close (" + ", ".join(detail) + ").",
                  f"Not a trade level by itself. If price drifts toward it, the day is behaving as the models expected."))
    fair = i.get("pair_fair")
    if fair is not None and lv["Stretched downside"] * 0.995 < fair < lv["Stretched upside"] * 1.005:
        L.append((fair, "FAIR VALUE (vs SPY)", "Where QQQ 'should' trade given SPY (cointegration model).",
                  "A slow magnet over days, not an intraday target."))
    return sorted(L, key=lambda t: -t[0])


# Historical behaviour of each level, QQQ 2018-2026 walk-forward backtest (daily bars)
LEVEL_HISTORY = {
    "UPSIDE LIMIT": "since 2018 price traded up to it on ~13% of days but closed above it on only ~7%",
    "DOWNSIDE LIMIT": "since 2018 price traded down to it on ~18% of days; when it broke, QQQ kept falling into the close slightly more often than it bounced",
    "EXTREME HIGH": "since 2018 QQQ closed above it on only ~0.3% of days",
    "EXTREME LOW": "since 2018 QQQ closed below it on ~1.2% of days (4x more often than above the extreme high)",
    "MOMENTUM TRIGGER UP": "a break through it fast tends to keep running (self-exciting buying)",
    "MOMENTUM TRIGGER DOWN": "a break through it fast tends to keep running (self-exciting selling)",
    "FAIR VALUE (vs SPY)": "acts as a slow magnet over days rather than a same-day target",
}
TOUCH_SCALE = 0.70                 # calibrates the reach-probability to the 13-18% touch rates seen in the backtest
SIDE_ADJ = {"down": 1.15, "up": 0.85}


def best_levels(c: Ctx, k: Composite, n: int = 3) -> List[dict]:
    """Rank confluence zones by (chance price trades there) x (how many independent methods agree) x (bias relevance)."""
    S, m = c.spot, max(k.move, 1e-6)
    d = desk_context(c, k)
    out = []
    for z in confluence_zones(c, k):
        x = abs(math.log(z["price"] / S))
        if x < 0.3 * m:
            continue
        side = "up" if z["price"] > S else "down"
        p_touch = min(0.95, 2 * (1 - float(Ncdf(x / (TOUCH_SCALE * m)))) * SIDE_ADJ[side])
        nf = len(z["families"])
        bias_w = 1.25 if (k.verdict == "UP" and side == "down") or (k.verdict == "DOWN" and side == "up") else 1.0
        why = f"{'under 1' if p_touch < 0.01 else f'{p_touch*100:.0f}'}% chance price trades here today"
        if nf >= 2:
            parts = []
            for fam in z["families"]:
                nm = ", ".join(x_["name"].lower() for x_ in z["items"] if x_["family"] == fam)
                parts.append(f"{fam} ({nm})")
            why += f"; {nf} independent methods agree: " + "; ".join(parts)
        if bias_w > 1:
            why += f"; it's where the {k.verdict.lower()} bias says to act"
        hist = "; ".join(LEVEL_HISTORY[x_["name"]] for x_ in z["items"] if x_["name"] in LEVEL_HISTORY)
        names = list(dict.fromkeys(z["names"]))
        rec = dict(price=z["price"], name=" + ".join(names), p=p_touch, families=z["families"],
                   score=p_touch * (1 + 0.6 * (nf - 1)) * bias_w, why=why, react=z["main"]["react"], history=hist)
        if z["hi"] - z["lo"] > 1e-9:
            rec["zone"] = (z["lo"], z["hi"])
        out.append(rec)
    out.sort(key=lambda r_: -r_["score"])
    picked = []
    for r_ in out:
        if len(picked) >= 2 and r_["p"] < 0.03:
            continue
        picked.append(r_)
        if len(picked) == n:
            break
    return picked


def exec_plan(c: Ctx, k: Composite, b: dict) -> List[str]:
    """Concrete 'how to take it' instructions for one top level: order type, trigger, stop, target."""
    S, m, i = c.spot, k.move, c.info
    lv = {nm: v for nm, v, _ in k.lv}
    ratio = c.fut_ratio if c.fut_ratio == c.fut_ratio else None

    def px(v):
        return f"{v:,.2f}" + (f" (NQ {v*ratio:,.0f})" if ratio else "")

    def pts(d):
        return f"{abs(d)*ratio:,.0f} NQ pts" if ratio else f"${abs(d):.2f}"
    name = b["name"]
    lo_, hi_ = b.get("zone", (b["price"], b["price"]))
    below = b["price"] < S                                   # level sits under the open
    bar = (i.get("hawkes_thr", 0.003) / 2) * S               # size of a normal 5-min candle in QQQ $
    if "FAIR VALUE" in name and "+" not in name:
        return ["Reference only - fair value pulls price over days, not minutes. No order here."]
    extreme = "EXTREME" in name
    edge = hi_ if below else lo_                             # first price you meet coming from the open
    far = lo_ if below else hi_
    d = desk_context(c, k)
    day = d["day_type"]
    if below:
        # ---- scenario A: stall / consolidation at support -> buy ----
        a_stop = far - 0.5 * m * S
        a_tgt = lv["Implied downside base"] if extreme else (lv["Implied upside base"] if k.verdict == "UP" else S)
        a_rr = abs(a_tgt - edge) / max(abs(edge - a_stop), 1e-9)
        a = (f"A) IF IT STALLS / CONSOLIDATES THERE -> LONG{' (small size)' if extreme else ''}: wait for price to reach {px(edge)}, "
             f"then for a 5-min candle to close back above it (or 2-3 small 5-min candles with no new low). "
             f"Then place a LIMIT buy at {px(edge)} for the retest. Stop {px(a_stop)} ({pts(edge - a_stop)} risk). "
             f"Target {px(a_tgt)} ({pts(a_tgt - edge)}, about 1:{a_rr:.1f}). If it never comes back to your limit, skip it.")
    else:
        # ---- upside: no shorts (shorting the upside limit lost 2,069 NQ pts in the 2025-26 backtest) ----
        a = (f"A) IF IT STALLS THERE -> TAKE PROFIT, DON'T SHORT: if you're long, take profits into {px(edge)}. "
             f"Don't open a short into upside levels: in the 2025-26 backtest, shorting the upside limit lost 2,069 NQ points (39% win rate).")
    lines = [a]
    if not extreme or not below:
        nxt = lv["Stretched downside"] if below else lv["Stretched upside"]
        if not extreme:
            b_trig = far - 0.1 * m * S if below else far + 0.1 * m * S
            b_stop = edge + 0.25 * m * S if below else edge - 0.25 * m * S
            b_rr = abs(nxt - b_trig) / max(abs(b_trig - b_stop), 1e-9)
            lines.append(
                f"B) IF IT BREAKS THROUGH FAST AND HOLDS -> {'SHORT' if below else 'LONG'}: a candle bigger than ~${2*bar:.2f} "
                f"through the level, then 2-3 five-minute candles holding {'below' if below else 'above'} it. Don't market-order the first "
                f"candle. Place a {'SELL-STOP' if below else 'BUY-STOP'} at {px(b_trig)}, just past the consolidation. "
                f"Stop back inside the level at {px(b_stop)} ({pts(b_trig - b_stop)} risk). Target {px(nxt)} ({pts(nxt - b_trig)}, about 1:{b_rr:.1f}).")
        if extreme:
            fav = "A - a 2-sigma stretch: take profits, don't chase and don't short"
        elif day.startswith("EXPANSION"):
            fav = f"B - it's an expansion/trend day ({d['day_why'][0]}), so breaks tend to run"
        elif below and (day.startswith("MEAN") or k.verdict == "UP"):
            fav = "A - " + ("dealers are long gamma, dips tend to get bought" if day.startswith("MEAN") else "the up bias says this level should hold")
        elif below and k.verdict == "DOWN":
            fav = "B - the down bias is pushing through this level"
        elif below:
            fav = "B slightly - historically breaks of the downside limit kept going more often than they bounced"
        elif k.verdict == "UP":
            fav = "B - the up bias is pushing through this level"
        else:
            fav = "A - take profits; QQQ closed above the upside limit on only ~7-11% of days"
        lines.append(f"Favoured today: {fav}. If neither happens (price just drifts through slowly), stand aside.")
    return lines


def how_to_enter(c: Ctx) -> List[str]:
    S, i, out = c.spot, c.info, []
    if "as_bid" in i:
        flow = "buy-heavy" if i["as_z"] > 1 else "sell-heavy" if i["as_z"] < -1 else "balanced"
        out.append(f"Use limit orders. Market makers would quote about {f(i['as_bid'])} bid / {f(i['as_ask'])} ask "
                   f"(Avellaneda-Stoikov model). Late order flow yesterday was {flow}.")
    if "hawkes_thr" in i:
        bar = i["hawkes_thr"] / 2 * S
        out.append(f"A normal 5-minute candle moves about ${bar:.2f}. A candle over ${2*bar:.2f} is a 'shock'. "
                   f"About {i['hawkes_rho']*100:.0f}% of shocks set off another one, so after a shock wait "
                   f"~{max(5, round(i['hawkes_mem']))} minutes for things to settle before entering.")
    if "rl_sched" in i:
        sl = [(n, u) for n, u in enumerate(i["rl_sched"]) if u > 0]
        t0 = datetime(2000, 1, 1, 9, 30)
        parts = ", ".join(f"{(t0+timedelta(minutes=6*n)).strftime('%H:%M')} {10*u}%" for n, u in sl)
        better = i["rl_cost"] < i["rl_twap"] - 0.05
        out.append(f"For a bigger position, split the order. The reinforcement-learning agent's plan (NY time): {parts}. "
                   + (f"Expected slippage {i['rl_cost']:.2f}bp vs {i['rl_twap']:.2f}bp for even slices."
                      if better else "Today, simple even slices over the hour work just as well."))
    return out


def market_story(c: Ctx, k: Composite) -> str:
    """Plain-English paragraph of what the model families say, keeping the key numbers."""
    i, S, bits = c.info, c.spot, []
    if "iv" in i:
        rich = "more" if i["vrp"] > 0.02 else "less" if i["vrp"] < -0.02 else "about as much"
        bits.append(f"Options traders are paying for {rich} movement than QQQ is actually making "
                    f"(implied {pc(i['iv'],0)} a year vs realised {pc(i['rv21'],0)}, Black-Scholes).")
    if "sabr_ratio" in i:
        r = i["sabr_ratio"]
        bits.append(f"Demand for crash protection is {'high' if r > 0.15 else 'low' if r < 0.05 else 'normal'} "
                    f"(SABR put skew {r:.2f})" + (", and the Heston model says selloffs tend to come with volatility spikes." if i.get('heston_rho', 0) < -0.3 else "."))
    if "garch_pctl" in i:
        busy = "calmer than usual" if i["garch_pctl"] < 0.35 else "busier than usual" if i["garch_pctl"] > 0.65 else "about average"
        bits.append(f"Today's expected swing is {busy}, {_ord(i['garch_pctl']*100)} percentile of the last few years (GARCH).")
    if "hmm_state" in i:
        st = i["hmm_state"]
        plain = ("a calm, steady phase" if "CALM" in st else "a high-volatility phase where moves overshoot" if "STRESS" in st
                 else "a trending bull phase" if st == "BULL" else "a normal phase")
        bits.append(f"The market-state model puts us in {plain} ({i['hmm_prob']*100:.0f}% confident, Hidden Markov).")
    if "pair_fair" in i:
        g = S / i["pair_fair"] - 1
        word = "stretched high" if g > 0.02 else "cheap" if g < -0.02 else "about fairly priced"
        bits.append(f"Against SPY, QQQ is {word} ({pc(g,1)} vs fair value {f(i['pair_fair'])}).")
    if "tail1" in i:
        bits.append(f"Risk check: {_one_in(i['tail1'])} chance of a 1% drop today, and on a bad day (1 in 20) "
                    f"expect to lose about {pc(i['var95'],1)} (Monte Carlo).")
    if "arima_fc" in i:
        bits.append(f"Pure price-pattern forecasts are near zero ({i['arima_fc']:+.2f}% from ARIMA), so don't expect yesterday's direction to carry over.")
    if "hawkes_ex_up" in i:
        side = "buying" if i["hawkes_ex_up"] > i["hawkes_ex_dn"] else "selling"
        bits.append(f"In the order flow, {side} bursts are the more active side going into the open (Hawkes).")
    return " ".join(bits)


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:] if t else t


def desk_note(c: Ctx, k: Composite) -> str:
    """One synthesis: the regime, how the options / volatility / structure / cross-asset reads confirm or contradict
    each other, and the stance that follows."""
    S, m, d, i = c.spot, max(k.move, 1e-6), desk_context(c, k), c.info
    lv = {nm: v for nm, v, _ in k.lv}
    g = d.get("gamma")
    out = []
    # 1. regime
    dt = d['day_type'].lower()
    out.append(f"Today reads as {'an' if dt[0] in 'aeiou' else 'a'} {dt}: " + ", ".join(d["day_why"]) + ".")
    # 2. volatility complex + options positioning, cross-checked
    vt = []
    if _ok(d["ts_back"]):
        vt.append(f"VIX term structure is {'in backwardation (front > back, a stress signal)' if d['ts_back'] > 1 else 'in contango (normal)'} "
                  f"at {d['ts_back']:.2f}" + (f", with 9-day vol {'above' if d['ts_front'] > 1 else 'below'} 30-day" if _ok(d["ts_front"]) else ""))
    if "iv" in i:
        vt.append(f"options are pricing {pc(i['iv'],0)} a year vs {pc(i['rv21'],0)} realised "
                  f"({'expensive' if i['vrp'] > 0.02 else 'cheap' if i['vrp'] < -0.02 else 'fair'})")
    if "garch_pctl" in i:
        vt.append(f"GARCH puts today's swing at the {_ord(i['garch_pctl']*100)} percentile")
    if vt:
        out.append("On volatility, " + "; ".join(vt) + ".")
    if g:
        gx = (f"Dealers are {'long' if g['gex'] > 0 else 'short'} gamma (about ${abs(g['gex'])/1e6:,.0f}M per 1% move)"
              + (f" with the flip at {f(g['flip'])}" if _ok(g.get("flip")) else "")
              + (", so they tend to dampen moves" if d["gamma_regime"] == "long" else ", so moves can feed on themselves" if d["gamma_regime"] == "short" else ""))
        walls = []
        ul, dl = lv["Implied upside base"], lv["Implied downside base"]
        if _ok(g.get("call_wall")):
            rel = "right at" if abs(g["call_wall"] - ul) < 0.15 * m * S else "inside" if g["call_wall"] < ul else "beyond"
            walls.append(f"the call wall at {f(g['call_wall'],0)} sits {rel} the upside limit ({f(ul)})")
        if _ok(g.get("put_wall")):
            rel = "right at" if abs(g["put_wall"] - dl) < 0.15 * m * S else "inside" if g["put_wall"] > dl else "beyond"
            walls.append(f"the put wall at {f(g['put_wall'],0)} sits {rel} the downside limit ({f(dl)})")
        txt = gx + (". " + _cap("; ".join(walls)) if walls else "")
        if any("right at" in w for w in walls):
            txt += ", so options positioning and the volatility model point to the same edges"
        elif any("inside" in w for w in walls):
            txt += ", so positioning may stop price before the statistical edge"
        out.append(txt + "." + (f" Put/call open interest is {g['pc_oi']:.2f}." if _ok(g.get("pc_oi")) else ""))
    else:
        out.append("Options positioning isn't available for this run (no live chain), so the levels rest on the volatility models and price structure.")
    # 3. trend / stretch vs the bias
    tr = (f"The trend is a {d['trend']}" + (f" (price {'above' if S > d['sma200'] else 'below'} the 200-day at {f(d['sma200'])}, "
                                               f"3-month momentum {pc(d['mom63'],1)})" if _ok(d["sma200"]) else "")
          + f", and price is {d['stretch']} (RSI(2) {d['rsi2']:.0f}, {d['z20']:+.1f} sd from the 20-day).")
    out.append(tr)
    # 4. cross-asset
    ca = []
    if _ok(d["credit5"]):
        ca.append(f"credit {'firming' if d['credit5'] > 0 else 'weakening'} (HYG vs LQD {pc(d['credit5'],1)} over 5 days)")
    if _ok(d["dxy5"]):
        ca.append(f"the dollar {'up' if d['dxy5'] > 0 else 'down'} {pc(abs(d['dxy5']),1)}")
    if _ok(d["y10_5d_bp"]):
        ca.append(f"10-year yields {d['y10_5d_bp']:+.0f}bp")
    if _ok(d["semis5"]):
        ca.append(f"semis {'leading' if d['semis5'] > 0 else 'lagging'} QQQ by {pc(abs(d['semis5']),1)}")
    if ca:
        out.append(f"Cross-asset reads {d['risk']}: " + ", ".join(ca) + ".")
    # 5. structure + calendar
    st = [f"QQQ opens {pc(d['gap'],2)} from yesterday's close ({abs(d['gap_z'])*100:.0f}% of a normal day)"]
    if _ok(d["pm_high"]):
        st.append(f"premarket ranged {f(d['pm_low'])}-{f(d['pm_high'])}")
    st.append(f"yesterday's range was {f(d['pdl'])}-{f(d['pdh'])}" + (" (NR7)" if d["nr7"] else ""))
    cal = (["it's OpEx day, so pinning near big strikes is common"] if d["opex_day"] else ["it's OpEx week"] if d["opex_week"] else [])
    cal += ["it's the turn of the month, when index inflows are typically strong"] if d["turn_of_month"] else []
    out.append(_cap("; ".join(st)) + "." + (" " + _cap("; ".join(cal)) + "." if cal else ""))
    # 6. how it all lines up
    if k.verdict in ("UP", "DOWN"):
        names_for = [n for n, v in d["ctx_votes"].items() if v == (1 if k.verdict == "UP" else -1)]
        names_vs = [n for n, v in d["ctx_votes"].items() if v == (-1 if k.verdict == "UP" else 1)]
        out.append(f"Putting it together: the models lean {k.verdict.lower()}"
                   + (f", supported by {', '.join(names_for)}" if names_for else "")
                   + (f" but contradicted by {', '.join(names_vs)}" if names_vs else "")
                   + f" - conviction {d['conviction']}. Stance: {d['stance']}.")
    else:
        lean = d["ctx_lean"]
        out.append(f"Putting it together: the models have no directional edge"
                   + (f"; the context leans {'up' if lean > 0 else 'down'} ({', '.join(n for n, v in d['ctx_votes'].items() if v)})" if lean else "; the context is split too")
                   + f". Stance: {d['stance']}.")
    return " ".join(out)


def actual_text(c: Ctx, k: Composite) -> List[str]:
    """For replays: how the day really played out against the levels."""
    a = c.info.get("actual")
    if not a:
        return []
    lv = {nm: v for nm, v, _ in k.lv}
    ul, dl, xh, xl = lv["Implied upside base"], lv["Implied downside base"], lv["Stretched upside"], lv["Stretched downside"]
    cl = a["close"]
    if cl > xh:
        where = "ABOVE the extreme high (a rare 2-sigma day)"
    elif cl > ul:
        where = "between the upside limit and the extreme high (outside the normal range)"
    elif cl >= dl:
        where = "INSIDE the normal range (range call correct)"
    elif cl >= xl:
        where = "between the downside limit and the extreme low (outside the normal range)"
    else:
        where = "BELOW the extreme low (a rare 2-sigma day)"
    up = cl > a["open"]
    if k.verdict == "UP":
        bias = "correct" if up else "wrong"
    elif k.verdict == "DOWN":
        bias = "correct" if not up else "wrong"
    else:
        bias = "no call made"
    touched = [nm for nm, v, hit in (("upside limit", ul, a["high"] >= ul), ("downside limit", dl, a["low"] <= dl),
                                       ("extreme high", xh, a["high"] >= xh), ("extreme low", xl, a["low"] <= xl)) if hit]
    return [
        f"Open {f(a['open'])}  High {f(a['high'])}  Low {f(a['low'])}  Close {f(cl)}  ({pc(cl/a['open']-1,2)} open to close)",
        f"Close finished {where}.",
        f"Bias said {k.verdict} -> QQQ closed {'up' if up else 'down'} from the open: {bias}.",
        "Levels traded through during the session: " + (", ".join(touched) if touched else "none (stayed inside both limits)"),
    ]


def summary_text(c: Ctx, k: Composite, res: List[Result]) -> str:
    S = c.spot
    L = [
        f"NY OPEN PLAYBOOK - {c.now.strftime('%a %d %b %Y')}",
        f"{c.cfg.equity} {S:,.2f}  (yesterday's close {c.prev_close:,.2f}, gap {pc(S/c.prev_close-1,2)})   "
        f"normal move today +/-{pc(k.move,2)} = +/-${S*k.move:,.2f}",
        "",
        "1. TODAY'S BIAS",
        bias_text(c, k, res),
        f"Day type: {desk_context(c, k)['day_type']}  ->  {desk_context(c, k)['stance']}"
        + (f"  (conviction {desk_context(c, k)['conviction']})" if k.verdict in ('UP', 'DOWN') else ""),
        "",
        "KEY ZONES TODAY  (context: where independent methods agree)",
    ]
    for i, b in enumerate(best_levels(c, k), 1):
        px_txt = f"{b['zone'][0]:,.2f}-{b['zone'][1]:,.2f} | NQ {fut(c, b['zone'][0])}-{fut(c, b['zone'][1])}" if "zone" in b else f"{b['price']:,.2f} | NQ {fut(c, b['price'])}"
        L.append(f"   #{i}  {px_txt}   {b['name']}")
        L.append(f"        Why: {b['why']}.")
        if b["history"]:
            L.append(f"        History: {b['history']}.")
    L += ["", "2. ALL LEVELS  (context, QQQ | NQ/MNQ)"]
    for v, nm, what, react in level_ladder(c, k):
        L.append(f"   {v:>9,.2f} | {fut(c, v):>8}   {nm}")
        L.append(f"        What it is: {what}")
        L.append(f"        If price gets here: {react}")
    L += ["", "3. WHAT NOT TO DO"] + [f" - {d}" for d in donts_list(c, k)]
    L += ["", "4. HOW TO ENTER"] + [f" - {e}" for e in how_to_enter(c)]
    L += ["", "5. DESK NOTE - HOW TODAY FITS TOGETHER", desk_note(c, k)]
    act = actual_text(c, k)
    if act:
        L += ["", "6. WHAT ACTUALLY HAPPENED (replay check)"] + [f" - {x}" for x in act]
    L += ["", "Model votes (weight x signal, for reference):"]
    for r in res:
        if r.ok and r.weight > 0:
            L.append(f"   {r.title:<44} w={r.weight:.1f}  signal={r.signal:+.2f}")
    bad = [r.title for r in res if not r.ok]
    if bad:
        L.append("Skipped: " + ", ".join(bad))
    for n in c.notes:
        L.append("Note: " + n)
    L.append("Statistical context only. Not financial advice.")
    return "\n".join(L)


def full_report(c: Ctx, k: Composite, res: List[Result]) -> str:
    out = ["=" * 100, "NY OPEN QUANT DESK - FULL REPORT", "=" * 100, summary_text(c, k, res), ""]
    sec = None
    for r in res:
        if r.section != sec:
            sec = r.section
            out += ["", "#" * 100, f"## {sec.upper()}", "#" * 100]
        out += ["", f"--- {r.title}  [{'ok' if r.ok else 'SKIPPED'}]  weight={r.weight:.1f} signal={r.signal:+.2f}", f"    {r.headline}"]
        out += ["    " + ln for ln in r.lines]
    return "\n".join(out)


# =============================================================================
# DISCORD
# =============================================================================
def _post(cfg: Config, payload: dict, files: Optional[dict] = None):
    for attempt in range(5):
        if files:
            resp = requests.post(cfg.webhook, data={"payload_json": json.dumps(payload)}, files=files, timeout=30)
        else:
            resp = requests.post(cfg.webhook, json=payload, timeout=30)
        if resp.status_code == 429:
            time.sleep(float(resp.json().get("retry_after", 1.5)) + 0.25)
            continue
        if resp.status_code >= 300:
            raise RuntimeError(f"Discord HTTP {resp.status_code}: {resp.text[:200]}")
        return
    raise RuntimeError("Discord rate limit: gave up")


def reversal_levels(c: Ctx, k: Composite, n: int = 2) -> List[dict]:
    """The n confluence zones most likely to produce a reversal today, with the reasons."""
    S, m, d = c.spot, max(k.move, 1e-6), desk_context(c, k)
    fam_w = {"options positioning": 0.5, "yesterday's structure": 0.3, "yesterday's volume": 0.3,
             "volatility model": 0.2, "overnight structure": 0.2, "trend averages": 0.15, "order flow (Hawkes)": 0.0, "relative value": 0.0}
    gam = 1.2 if d["gamma_regime"] == "long" else 0.8 if d["gamma_regime"] == "short" else 1.0
    out = []
    for z in confluence_zones(c, k):
        x = abs(math.log(z["price"] / S))
        if x < 0.3 * m or x > 2.2 * m:
            continue
        below = z["price"] < S
        p_touch = min(0.95, 2 * (1 - float(Ncdf(x / (TOUCH_SCALE * m)))) * SIDE_ADJ["down" if below else "up"])
        fams = z["families"]
        # confluence matters more than proximity: sqrt dampens the reach probability, agreement is weighted up
        score = math.sqrt(p_touch) * (1 + 0.8 * (len(fams) - 1) + sum(fam_w.get(f_, 0) for f_ in fams)) * (1.0 if below else 0.7) * gam
        out.append(dict(price=z["price"], below=below, p=p_touch, fams=fams, names=list(dict.fromkeys(z["names"])), score=score))
    out.sort(key=lambda r_: -r_["score"])
    multi = [r_ for r_ in out if len(r_["fams"]) >= 2 and r_["p"] >= 0.15]
    if len(multi) >= n:
        out = multi                                     # when real confluence zones are in reach, use only those
    for r_ in out:
        r_["strength"] = "strong" if r_["score"] >= 1.2 else "medium" if r_["score"] >= 0.7 else "weak"
    return out[:n]


def compact_lines(c: Ctx, k: Composite, res: List[Result]) -> List[str]:
    """Bias, range, fair value and the 2 best reversal levels with execution - nothing else."""
    S, d, cfg = c.spot, desk_context(c, k), c.cfg
    lv = {nm: v for nm, v, _ in k.lv}
    nq = lambda x: f"{x:,.2f} (NQ {fut(c, x)})"           # every level: QQQ price, NQ equivalent in brackets
    ratio = c.fut_ratio if c.fut_ratio == c.fut_ratio else 41.0
    q = lambda pts: pts / ratio                       # NQ points -> QQQ dollars
    L = []
    gap = S / c.prev_close - 1
    L.append(f"**NY Open · {c.now.strftime('%a %d %b')}**" + ("  ·  REPLAY" if cfg.as_of else "")
             + f"  ·  QQQ {nq(S)}  ·  gap {pc(gap,2)}")
    # ---- bias
    bias = {"UP": "LEAN UP", "DOWN": "LEAN DOWN"}.get(k.verdict, "NO CLEAR BIAS")
    conv = f", conviction {d['conviction']}" if k.verdict in ("UP", "DOWN") else ""
    L.append(f"**Bias:** {bias} - {k.p_up*100:.0f}% chance of closing above the open{conv} · "
             f"{d['day_type'].split(' /')[0].lower()} day ({d['day_why'][0].split(' (')[0]})")
    # ---- range
    L.append(f"**Range:** {nq(lv['Implied downside base'])} – {nq(lv['Implied upside base'])} · ±{pc(k.move,2)} · ~78% of days close inside · "
             f"extremes {nq(lv['Stretched downside'])} / {nq(lv['Stretched upside'])}, <3% of days close beyond")
    # ---- fair value
    fv = []
    fair = c.info.get("pair_fair")
    if _ok(fair):
        fv.append(f"{nq(fair)} vs SPY ({'price above value' if S > fair else 'price below value'}, {pc(abs(S/fair-1),1)} away)")
    g = d.get("gamma")
    if g and _ok(g.get("max_pain")):
        fv.append(f"options pin (max pain) {nq(g['max_pain'])}")
    if g and _ok(g.get("flip")):
        fv.append(f"gamma flip {nq(g['flip'])}")
    L.append("**Fair value:** " + (" · ".join(fv) if fv else "n/a today"))
    # ---- the 2 reversal levels
    calm = d["gamma_regime"] == "long" or d["day_type"].startswith("MEAN")
    plans = []
    for i_, r in enumerate(reversal_levels(c, k), 1):
        lvl = r["price"]
        nmf = len(r["fams"])
        short_f = {"order flow (Hawkes)": "Hawkes order flow"}
        fam_txt = ", ".join(short_f.get(f_, f_) for f_ in r["fams"][:3])
        side = "BUY" if r["below"] else "SELL"
        L.append(f"**{i_}. {side} {nq(lvl)}** - {' + '.join(x.title() for x in r['names'][:2])}: "
                 + (f"{nmf} independent methods agree ({fam_txt})" if nmf > 1 else f"{fam_txt}")
                 + f", {r['p']*100:.0f}% chance price reaches it · strength: {r['strength']}")
        # every instruction carries exact NQ prices: entry, stop, TP1 (back to the open), TP2 (150 pts)
        stop_d, tgt_d = q(cfg.stop_pts), q(cfg.target_pts)
        if r["below"]:
            if calm or nmf >= 3:
                entry = lvl + q(2)
                sl, tp2 = lvl - stop_d, entry + tgt_d
                how = (f"LIMIT BUY {nq(entry)} (place it before price arrives) · SL {nq(sl)} · TP1 {nq(S)} · TP2 {nq(tp2)}")
            else:
                entry = lvl
                sl, tp2 = lvl - stop_d, lvl + tgt_d
                how = (f"Wait for a 5-min candle to CLOSE back above {nq(lvl)} → MARKET BUY ≈{nq(entry)} · SL {nq(sl)} · "
                       f"TP1 {nq(S)} · TP2 {nq(tp2)} (skip if that candle's low is already below {nq(sl)})")
        else:
            entry = lvl
            sl, tp2 = lvl + stop_d, lvl - tgt_d
            how = (f"Wait for a 5-min candle to CLOSE back below {nq(lvl)} → MARKET SELL ≈{nq(entry)}, small size · SL {nq(sl)} · "
                   f"TP1 {nq(S)} · TP2 {nq(tp2)} (skip if that candle's high is already above {nq(sl)})")
        use_tp1 = abs(entry - S) * ratio >= cfg.stop_pts
        if not use_tp1:                                               # open too close to be a sensible first target
            how = how.replace(f"TP1 {nq(S)} · ", "")
        plans.append(dict(n=i_, side=1 if r["below"] else -1, level=lvl, entry=entry, sl=sl, tp1=S if use_tp1 else None, tp2=tp2,
                          limit=how.startswith("LIMIT")))
        L.append(f"   ↳ {how}")
    c.info["plans"] = plans
    if cfg.as_of:
        L += replay_check(c, k, plans)
    return L[:15]


def _fetch_day_1m(cfg: Config, day) -> Optional[pd.DataFrame]:
    """Regular-session 1-minute QQQ bars for a past day (Yahoo keeps about 30 days of 1-minute data)."""
    q = _flat(yf.download(cfg.equity, start=day.isoformat(), end=(day + timedelta(days=1)).isoformat(),
                          interval="1m", auto_adjust=False, progress=False)).rename(columns=str.lower)
    if q.empty:
        return None
    if q.index.tz is None:
        q.index = q.index.tz_localize("UTC")
    q.index = q.index.tz_convert(ET)
    return q.between_time("09:30", "15:59")


def simulate_plan(pl: dict, bars: pd.DataFrame, ratio: float) -> dict:
    """Play one card instruction out on real intraday bars (1- or 5-minute, regular session, ET index).
    Limit: fills when price trades to the entry. Confirmation: price must break the level, then a 5-min candle must
    close back through it -> market entry at that close (skipped if that candle already went past the stop).
    Exits: stop first if stop and target fall inside one bar; TP1 (the open) only counts if it's on the profit side
    of the fill; TP2 or the stop ends the trade; anything open is closed at the last bar of the session."""
    sd, L = pl["side"], pl["level"]
    out = dict(status="no fill", entry=np.nan, entry_t=None, tp1_hit=False, tp1_t=None, exit_reason=None,
               exit_px=np.nan, exit_t=None, pts_full=0.0, pts_split=0.0, risk_pts=np.nan)
    if bars is None or len(bars) < 10:
        out["status"] = "no data"
        return out
    hi, lo, cl = bars["high"].values, bars["low"].values, bars["close"].values
    tm = np.array([x.hour * 60 + x.minute for x in bars.index])
    i0, entry = None, pl["entry"]
    if pl["limit"]:
        hit = np.where((tm <= 15 * 60 + 30) & ((lo <= entry) if sd > 0 else (hi >= entry)))[0]
        if len(hit):
            i0 = int(hit[0])
    else:
        c5 = bars["close"].resample("5min").last().dropna()
        l5, h5 = bars["low"].resample("5min").min(), bars["high"].resample("5min").max()
        broke = False
        for t5, cc in c5.items():
            if t5.hour * 60 + t5.minute > 15 * 60 + 25:
                break
            lo5, hi5 = l5[t5], h5[t5]
            if not broke and ((sd > 0 and lo5 < L) or (sd < 0 and hi5 > L)):
                broke = True
            if broke and ((sd > 0 and cc > L) or (sd < 0 and cc < L)):
                if (sd > 0 and lo5 <= pl["sl"]) or (sd < 0 and hi5 >= pl["sl"]):
                    out.update(status="skipped", entry_t=t5 + timedelta(minutes=5))
                    return out
                j = int(np.searchsorted(bars.index.values, (t5 + timedelta(minutes=5)).to_datetime64()))
                if j < len(bars):
                    i0, entry = j, float(cc)
                break
    if i0 is None:
        return out
    sl, tp1, tp2 = pl["sl"], pl["tp1"], pl["tp2"]
    if tp1 is not None and sd * (tp1 - entry) <= 0:
        tp1 = None
    out.update(status="filled", entry=entry, entry_t=bars.index[i0], risk_pts=sd * (entry - sl) * ratio)
    exit_px, reason, exit_i = None, None, None
    for j in range(i0, len(bars)):
        if (sd > 0 and lo[j] <= sl) or (sd < 0 and hi[j] >= sl):
            exit_px, reason, exit_i = sl, "STOP", j
            break
        if j > i0 and tp1 is not None and not out["tp1_hit"] and ((sd > 0 and hi[j] >= tp1) or (sd < 0 and lo[j] <= tp1)):
            out.update(tp1_hit=True, tp1_t=bars.index[j])
        if j > i0 and ((sd > 0 and hi[j] >= tp2) or (sd < 0 and lo[j] <= tp2)):
            exit_px, reason, exit_i = tp2, "TP2", j
            break
    if exit_px is None:
        exit_px, reason, exit_i = float(cl[-1]), "CLOSE", len(bars) - 1
    full = sd * (exit_px - entry) * ratio
    split = 0.5 * sd * (tp1 - entry) * ratio + 0.5 * full if out["tp1_hit"] else full
    out.update(exit_reason=reason, exit_px=exit_px, exit_t=bars.index[exit_i], pts_full=full, pts_split=split)
    return out


def replay_check(c: Ctx, k: Composite, plans: List[dict], bars: Optional[pd.DataFrame] = None) -> List[str]:
    """Play each instruction out on the real 1-minute bars of the replayed day."""
    day = c.now.date()
    nq = lambda x: f"{x:,.2f} (NQ {fut(c, x)})"
    ratio = c.fut_ratio if c.fut_ratio == c.fut_ratio else 41.0
    try:
        bars = bars if bars is not None else _fetch_day_1m(c.cfg, day)
    except Exception as e:
        bars = None
        log.warning("replay 1m fetch failed: %s", e)
    a = c.info.get("actual")
    out = []
    if a:
        lv = {nm: v for nm, v, _ in k.lv}
        inside = lv["Implied downside base"] <= a["close"] <= lv["Implied upside base"]
        out.append(f"**What happened:** high {nq(a['high'])} · low {nq(a['low'])} · close {nq(a['close'])} → "
                   f"{'closed INSIDE the range' if inside else 'closed OUTSIDE the range'}")
    if bars is None or len(bars) < 30:
        out.append("(1-minute data for this day isn't available - Yahoo only keeps about 30 days - so the trades can't be checked)")
        return out
    for pl in plans:
        r = simulate_plan(pl, bars, ratio)
        name = f"{pl['n']}. {'BUY' if pl['side'] > 0 else 'SELL'} {nq(pl['level'])}"
        if r["status"] == "skipped":
            out.append(f"{name}: confirmation candle went past the stop at {r['entry_t'].strftime('%H:%M')} → skipped (rule)")
        elif r["status"] != "filled":
            out.append(f"{name}: never triggered (price {'did not reach' if pl['limit'] else 'did not break and reclaim'} the level)")
        else:
            steps = [f"filled ≈{nq(r['entry'])} {r['entry_t'].strftime('%H:%M')}"]
            if r["tp1_hit"]:
                steps.append(f"TP1 {r['tp1_t'].strftime('%H:%M')}")
            ex = {"STOP": "stopped", "TP2": "TP2 hit", "CLOSE": "closed at the bell"}[r["exit_reason"]]
            steps.append(f"{ex} {nq(r['exit_px'])} {r['exit_t'].strftime('%H:%M')}")
            out.append(f"{name}: " + " → ".join(steps) + f" = {r['pts_full']:+.0f} pts (half off at TP1: {r['pts_split']:+.0f})")
    return out


def send_discord(cfg: Config, c: Ctx, k: Composite, res: List[Result], report: str):
    """One compact card (<= 15 lines). The full report rides along as an attachment."""
    color = 0xE74C3C if k.verdict == "DOWN" else 0x2ECC71 if k.verdict == "UP" else 0x95A5A6
    lines = compact_lines(c, k, res)
    card = {"description": "\n".join(lines)[:4000], "color": color,
            "footer": {"text": "Full report attached · statistical context, not financial advice" + (" · DEMO DATA" if cfg.demo else "")}}
    _post(cfg, {"username": "NY Open Quant Desk", "embeds": [card]},
          files={"files[0]": ("full_report.txt", report.encode("utf-8"), "text/plain")})


# =============================================================================
# RUN / SCHEDULE
# =============================================================================
def run_once(cfg: Config):
    log.info("building context (%s)...", "demo" if cfg.demo else "live")
    c = make_demo(cfg) if cfg.demo else load_live(cfg)
    c = finish_ctx(c)
    res = run_all(c)
    k = compose(res, c)
    report = full_report(c, k, res)
    if cfg.dry_run or not cfg.webhook:
        if not cfg.dry_run:
            log.warning("DISCORD_WEBHOOK_URL not set -> printing instead of sending")
        print(report)
        return
    send_discord(cfg, c, k, res, report)
    log.info("sent to Discord: %s (score %+.2f)", k.verdict, k.score)


def _nyse_holidays(year: int) -> set:
    """Full-day NYSE holidays (built-in backup when pandas_market_calendars is missing)."""
    from datetime import date as D

    def nth(month, weekday, n):                      # n-th weekday of month (n=-1 -> last)
        if n > 0:
            d = D(year, month, 1)
            d += timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
            return d
        d = D(year, month + 1, 1) - timedelta(days=1) if month < 12 else D(year, 12, 31)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    def observed(d):
        return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d
    a, b, c = year % 19, year // 100, year % 100        # Easter (anonymous Gregorian algorithm)
    dd, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - dd - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    mm = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * mm + 90) // 25
    day = (h + l - 7 * mm + 33 * month + 19) % 32
    easter = D(year, month, day)
    hol = {nth(1, 0, 3), nth(2, 0, 3), easter - timedelta(days=2), nth(5, 0, -1), nth(9, 0, 1), nth(11, 3, 4),
           observed(D(year, 7, 4)), observed(D(year, 12, 25))}
    ny = D(year, 1, 1)
    if ny.weekday() != 5:                            # NYSE does not close Fri 31 Dec for a Saturday New Year
        hol.add(observed(ny))
    if year >= 2022:
        hol.add(observed(D(year, 6, 19)))
    return hol


def _is_trading_day(d) -> bool:
    if d.weekday() >= 5:
        return False
    pmc = _try("pandas_market_calendars")
    if pmc is not None:
        try:
            return len(pmc.get_calendar("NYSE").schedule(start_date=d, end_date=d)) > 0
        except Exception:
            pass
    return d not in _nyse_holidays(d.year)


def _next_trigger(cfg: Config, now: datetime) -> datetime:
    d = now.date()
    while True:
        t = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(seconds=cfg.run_delay_sec)
        if t > now and _is_trading_day(d):
            return t
        d += timedelta(days=1)


def scheduler(cfg: Config):
    log.info("scheduler started - fires %ds after every NY open (09:30 ET, trading days)", cfg.run_delay_sec)
    while True:
        now = datetime.now(ET)
        t = _next_trigger(cfg, now)
        log.info("next run: %s", t.strftime("%a %Y-%m-%d %H:%M:%S %Z"))
        while datetime.now(ET) < t:
            time.sleep(min(30.0, max(0.5, (t - datetime.now(ET)).total_seconds())))
        try:
            run_once(cfg)
        except Exception:
            log.error("run failed:\n%s", traceback.format_exc())
            try:
                if cfg.webhook:
                    _post(cfg, {"content": "NY open quant run failed today - check the console logs."})
            except Exception:
                pass
        time.sleep(90)


def once_today(cfg: Config, latest_et=(11, 30), retries=3, earliest_et=None, use_stamp=True):
    """For a daily scheduler (launchd/cron): wait for today's NY open, post once, exit.
    Skips weekends/holidays, days already posted, and runs started after `latest_et`."""
    state = Path(os.getenv("NYQ_HOME", Path.home() / "ny_open_quant"))
    state.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ET)
    today = now.date()
    if not _is_trading_day(today):
        log.info("%s is not a NYSE trading day - nothing to do", today)
        return
    stamp = state / f"posted_{today.isoformat()}.ok"
    if earliest_et is not None and now < datetime(today.year, today.month, today.day, *earliest_et, tzinfo=ET):
        log.info("%s ET is too early - this is the other scheduled slot, exiting", now.strftime("%H:%M"))
        return
    if use_stamp and stamp.exists():
        log.info("already posted today (%s) - nothing to do", stamp.name)
        return
    cutoff = datetime(today.year, today.month, today.day, *latest_et, tzinfo=ET)
    if now > cutoff:
        log.warning("started at %s ET, after the %02d:%02d cutoff - skipping today", now.strftime("%H:%M"), *latest_et)
        return
    t = datetime(today.year, today.month, today.day, 9, 30, tzinfo=ET) + timedelta(seconds=cfg.run_delay_sec)
    if now < t:
        log.info("waiting until %s ET for the open", t.strftime("%H:%M:%S"))
        while datetime.now(ET) < t:
            time.sleep(min(30.0, max(0.5, (t - datetime.now(ET)).total_seconds())))
    for attempt in range(1, retries + 1):
        try:
            run_once(cfg)
            if not use_stamp:
                return
            stamp.write_text(datetime.now(ET).isoformat())
            for old in sorted(state.glob("posted_*.ok"))[:-30]:      # keep the folder tidy
                old.unlink(missing_ok=True)
            return
        except Exception:
            log.error("attempt %d/%d failed:\n%s", attempt, retries, traceback.format_exc())
            if attempt < retries:
                time.sleep(120)
    try:
        if cfg.webhook:
            _post(cfg, {"content": "NY open quant run failed today after 3 attempts - check ~/ny_open_quant/logs."})
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="NY-open quant desk -> Discord")
    ap.add_argument("--now", action="store_true", help="run once immediately instead of waiting for the NY open")
    ap.add_argument("--cloud", action="store_true", help="for GitHub Actions: post only if started 08:50-10:00 NY time, then exit")
    ap.add_argument("--once-today", action="store_true", help="for a daily scheduler: wait for today's open, post once, exit")
    ap.add_argument("--dry-run", action="store_true", help="print the report instead of posting to Discord")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no internet needed)")
    ap.add_argument("--ppo", action="store_true", help="also train a PPO execution agent (needs gymnasium + stable-baselines3)")
    ap.add_argument("--webhook", default=None, help="Discord webhook URL (or set DISCORD_WEBHOOK_URL)")
    ap.add_argument("--equity", default=None, help="ticker to model (default QQQ)")
    ap.add_argument("--date", default=None, help="replay a past open, e.g. --date 2026-08-18 (implies --now)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config(demo=a.demo, dry_run=a.dry_run, ppo=a.ppo)
    if a.webhook:
        cfg.webhook = a.webhook
    if a.equity:
        cfg.equity = a.equity
    if a.date:
        datetime.strptime(a.date, "%Y-%m-%d")      # validates the format
        cfg.as_of = a.date
        a.now = True
    if a.cloud:
        if not cfg.webhook and not cfg.dry_run:
            sys.exit("DISCORD_WEBHOOK_URL secret is not set in the GitHub repo")
        # Two cron slots run each weekday (13:20 and 14:20 UTC); exactly one lands in this window.
        once_today(cfg, latest_et=(10, 0), earliest_et=(8, 50), use_stamp=False)
    elif a.once_today:
        if not cfg.webhook and not cfg.dry_run:
            sys.exit("DISCORD_WEBHOOK_URL is not set (check ~/ny_open_quant/config.sh)")
        once_today(cfg)
    elif a.now or a.demo:
        run_once(cfg)
    else:
        if not cfg.webhook:
            sys.exit("Set DISCORD_WEBHOOK_URL (or pass --webhook) before running the scheduler.")
        scheduler(cfg)


if __name__ == "__main__":
    main()
