#!/usr/bin/env python3
"""
Full backtest of the NY Open Quant Desk morning card.

For every trading day in [start, end] it rebuilds exactly what the bot knew before the 09:30 open (history up to the
prior close + that day's official open), runs the bot's own code, builds the morning card, and then checks it:

  1) THE RANGE      - did QQQ close inside the range / extremes as often as the card says?
  2) THE BIAS       - was "LEAN UP / LEAN DOWN" right more often than just guessing up?
  3) THE 2 LEVELS   - each reversal level traded exactly as the card instructs (limit or 5-min confirmation, stop 35 pts,
                      TP1 = the open, TP2 = +150 pts), on real intraday bars, using the same simulator as the replays.
                      Two ways of managing it are scored: hold the whole position, or take half off at TP1.

Intraday bars: 2 years of QQQ 1-minute data from a public GitHub dataset (Aug-2024 -> Aug-2026) plus Yahoo's last
60 days of 5-minute bars. Days with neither are still scored for range and bias.
Not available historically: option chains (no call/put walls, gamma flip or max pain in the backtest - live mornings
have them, so live levels can differ). Not financial advice; past results do not guarantee future results.

    python backtest_full.py --start 2024-09-01                 # every day with intraday data
    python backtest_full.py --start 2025-01-01 --discord       # also post the summary to Discord
Results: backtest_days.csv (one row per day), backtest_trades.csv (one row per level), backtest_summary.txt
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

import ny_open_quant as q

SKIP = {"mc", "fdm", "root", "lattice", "rl"}          # weight 0: never change the bias, range or levels
MINUTE_URL = "https://raw.githubusercontent.com/ramires666/rtharb/main/data_cache/QQQ_1m.parquet"
OUT_DAYS, OUT_TRADES, OUT_TXT = "backtest_days.csv", "backtest_trades.csv", "backtest_summary.txt"
log = logging.getLogger("backtest")


# ------------------------------------------------------------------------------------------------- data
def download_daily(cfg, start, end):
    tickers = list(dict.fromkeys([cfg.equity, cfg.pair, "IWM", "TLT", "GLD", "IWD", "IWF", "MTUM", "^VIX", "^IRX",
                                  "^FVX", "^TNX", "^TYX", cfg.futures] + q.EXTRA_TICKERS))
    s = (start - timedelta(days=int(365.25 * cfg.years) + 15)).isoformat()
    e = (end + timedelta(days=3)).isoformat()
    log.info("downloading daily history %s -> %s for %d tickers", s, e, len(tickers))
    return q.yf.download(tickers, start=s, end=e, interval="1d", auto_adjust=True, progress=False, group_by="column")


def download_intraday(cfg):
    """{date: regular-session bars} from the 2-year 1-minute dataset, topped up with Yahoo's last 60 days of 5-min bars."""
    by_day = {}
    try:
        log.info("downloading 2-year QQQ 1-minute dataset ...")
        r = requests.get(MINUTE_URL, timeout=180)
        r.raise_for_status()
        m = pd.read_parquet(io.BytesIO(r.content))
        m.index = m.index.tz_convert(q.ET) if m.index.tz is not None else m.index.tz_localize(q.ET)
        m = m.rename(columns=str.lower).between_time("09:30", "15:59")
        for d, g in m.groupby(m.index.date):
            if len(g) >= 300:
                by_day[d] = g[["open", "high", "low", "close", "volume"]]
        log.info("1-minute data: %d sessions", len(by_day))
    except Exception as ex:
        log.warning("1-minute dataset unavailable (%s) - using Yahoo 5-minute data only", ex)
    try:
        y = q._flat(q.yf.download(cfg.equity, period="60d", interval="5m", auto_adjust=False, progress=False)).rename(columns=str.lower)
        if y.empty or not isinstance(y.index, pd.DatetimeIndex):
            log.info("Yahoo returned no 5-minute data")
            return by_day
        y.index = y.index.tz_convert(q.ET) if y.index.tz is not None else y.index.tz_localize("UTC").tz_convert(q.ET)
        y = y.between_time("09:30", "15:59")
        added = 0
        for d, g in y.groupby(y.index.date):
            if d not in by_day and len(g) >= 60:
                by_day[d] = g[["open", "high", "low", "close", "volume"]]
                added += 1
        log.info("Yahoo 5-minute data added %d more sessions", added)
    except Exception as ex:
        log.warning("Yahoo 5-minute download failed: %s", ex)
    return by_day


def context_for(cfg, raw, day):
    """Exactly what load_live() builds before the open, from pre-downloaded data (no option chain historically)."""
    now = datetime(day.year, day.month, day.day, 9, 30, 20, tzinfo=q.ET)
    lo = day - timedelta(days=int(365.25 * cfg.years))
    hist = raw[(raw.index.date < day) & (raw.index.date >= lo)]
    close = hist["Close"].copy().ffill()
    ohlc = pd.DataFrame({k.lower(): hist[k][cfg.equity] for k in ["Open", "High", "Low", "Close", "Volume"]}).dropna()
    px = close.reindex(ohlc.index).ffill()
    today = raw[raw.index.date == day]
    spot = float(today["Open"][cfg.equity].iloc[0])
    actual = {k: float(today[k.capitalize()][cfg.equity].iloc[0]) for k in ("open", "high", "low", "close")}
    fut_ratio = float("nan")
    try:
        fut_ratio = float(px[cfg.futures].dropna().iloc[-1]) / float(px[cfg.equity].iloc[-1])
    except Exception:
        pass
    rf = float(px["^IRX"].dropna().iloc[-1]) / 100.0 if "^IRX" in px else 0.04
    c = q.Ctx(cfg=cfg, now=now, px=px, ohlc=ohlc, intra=pd.DataFrame(), spot=spot, fut_ratio=fut_ratio, rf=rf, q=cfg.div_yield)
    c.info["actual"] = actual
    return q.finish_ctx(c)


# ------------------------------------------------------------------------------------------------- one day
def run_day(cfg, raw, intraday, day):
    c = context_for(cfg, raw, day)
    res = q.run_all(c)
    k = q.compose(res, c)
    q.compact_lines(c, k, res)                       # builds the card and stores today's 2 plans in c.info["plans"]
    lv = {nm: v for nm, v, _ in k.lv}
    a = c.info["actual"]
    dk = q.desk_context(c, k)
    ratio = c.fut_ratio if c.fut_ratio == c.fut_ratio else 41.0
    row = dict(date=day.isoformat(), verdict=k.verdict, score=k.score, p_up=k.p_up, move=k.move,
               open=a["open"], high=a["high"], low=a["low"], close=a["close"], prev_close=c.prev_close, ratio=ratio,
               XH=lv["Stretched upside"], UL=lv["Implied upside base"], DL=lv["Implied downside base"], XL=lv["Stretched downside"],
               fair=c.info.get("pair_fair", np.nan), day_type=dk["day_type"], vol_regime=dk["vol_regime"], trend=dk["trend"],
               conviction=dk["conviction"], has_intraday=day in intraday)
    for r in res:
        row[f"sig_{r.key}"] = r.signal if r.ok else np.nan
    trades = []
    bars = intraday.get(day)
    for pl in c.info.get("plans", []):
        sim = q.simulate_plan(pl, bars, ratio)
        trades.append(dict(date=day.isoformat(), rank=pl["n"], side="BUY" if pl["side"] > 0 else "SELL",
                           entry_type="LIMIT" if pl["limit"] else "CONFIRMATION", level=pl["level"],
                           level_nq=pl["level"] * ratio, dist_pts=abs(pl["level"] - a["open"]) * ratio,
                           day_type=dk["day_type"], verdict=k.verdict, status=sim["status"],
                           exit_reason=sim["exit_reason"], tp1_hit=sim["tp1_hit"], risk_pts=sim["risk_pts"],
                           pts_full=sim["pts_full"] - (0.75 if sim["status"] == "filled" else 0),
                           pts_split=sim["pts_split"] - (0.75 if sim["status"] == "filled" else 0)))
    return row, trades


# ------------------------------------------------------------------------------------------------- summary
def _stats(p):
    p = np.asarray(p, float)
    if len(p) == 0:
        return "no trades"
    w, l = p[p > 0], p[p <= 0]
    rr = w.mean() / -l.mean() if len(w) and len(l) and l.mean() < 0 else float("nan")
    t = p.mean() / (p.std(ddof=1) / math.sqrt(len(p))) if len(p) > 2 and p.std() > 0 else float("nan")
    return f"{len(p):>4} trades  win {100*(p > 0).mean():5.1f}%  avg R:R {rr:4.2f}  {p.mean():+6.1f} pts/trade  total {p.sum():+7.0f}  t {t:+5.2f}"


def summarise(D: pd.DataFrame, T: pd.DataFrame) -> str:
    L = []
    O, H, Lo, C = D.open, D.high, D.low, D.close
    UL, DL, XH, XL = D.UL, D.DL, D.XH, D.XL
    oc = np.log(C / O)
    up = oc > 0
    L += ["=" * 100, f"NY OPEN QUANT DESK - MORNING CARD BACKTEST   {D.date.iloc[0]} -> {D.date.iloc[-1]}   {len(D)} trading days", "=" * 100]

    L += ["", "1) THE RANGE", f"   {'what the card says':<52}{'claimed':>9}{'actual':>9}"]
    for lab, cl, act in [
        ("close inside the range", "~78%", ((C <= UL) & (C >= DL)).mean()),
        ("close inside the extremes", ">97%", ((C <= XH) & (C >= XL)).mean()),
        ("close above the range", "", (C > UL).mean()), ("close below the range", "", (C < DL).mean()),
        ("price trades through the top of the range", "", (H >= UL).mean()),
        ("price trades through the bottom of the range", "", (Lo <= DL).mean()),
    ]:
        L.append(f"   {lab:<52}{cl:>9}{100*act:>8.1f}%")

    L += ["", "2) THE BIAS", f"   base rate: QQQ closed above its open on {100*up.mean():.1f}% of days"]
    for v in ("UP", "DOWN", "NEUTRAL / NO EDGE"):
        s = D.verdict == v
        if s.any():
            r = oc[s]
            right = (r > 0).mean() if v == "UP" else (r < 0).mean() if v == "DOWN" else float("nan")
            L.append(f"   {v:<18}{s.sum():>5} days  closed up {100*(r > 0).mean():5.1f}%  avg open->close {1e4*r.mean():+6.1f}bp"
                     + (f"  -> bias right {100*right:.1f}%" if v != "NEUTRAL / NO EDGE" else ""))
    called = D.verdict != "NEUTRAL / NO EDGE"
    if called.any():
        L.append(f"   overall: right on {100*(np.sign(D.score[called]) == np.sign(oc[called])).mean():.1f}% of days with a call "
                 f"(always guessing UP: {100*up[called].mean():.1f}%)")

    L += ["", "3) THE 2 REVERSAL LEVELS  (traded exactly as the card says; NQ points after 0.75 pt cost; 1 contract)"]
    if T.empty:
        L.append("   no levels")
    else:
        Tt = T[T.status != "no data"]
        L.append(f"   days with intraday data: {Tt.date.nunique()}   levels: {len(Tt)}   "
                 f"filled {100*(Tt.status == 'filled').mean():.0f}%  ·  never triggered {100*(Tt.status == 'no fill').mean():.0f}%  ·  "
                 f"skipped by the stop rule {100*(Tt.status == 'skipped').mean():.0f}%")
        F = Tt[Tt.status == "filled"]
        if F.empty:
            L.append("   no filled trades")
        else:
            L.append(f"   {'':<34}hold the whole position to TP2 / stop / close")
            L.append(f"   {'ALL TRADES':<34}{_stats(F.pts_full)}")
            L.append(f"   {'':<34}take half off at TP1, rest to TP2 / stop / close")
            L.append(f"   {'ALL TRADES':<34}{_stats(F.pts_split)}")
            L.append(f"   exits: {F.exit_reason.value_counts().to_dict()}  ·  TP1 reached on {100*F.tp1_hit.mean():.0f}% of fills  ·  "
                     f"average real risk {F.risk_pts.mean():.0f} pts (confirmation fills can be further than 35 pts from the stop)")
            for col, lab in (("rank", "level"), ("side", "side"), ("entry_type", "entry"), ("day_type", "day type"), ("verdict", "bias")):
                L.append(f"   by {lab}:  (whole position | half off at TP1)")
                for v, g in F.groupby(col):
                    L.append(f"     {str(v)[:30]:<30}{_stats(g.pts_full)}  |  {g.pts_split.mean():+6.1f} pts/trade")
            F2 = F.assign(month=pd.to_datetime(F.date).dt.to_period("Q").astype(str))
            L.append("   by quarter (whole position):")
            for v, g in F2.groupby("month"):
                L.append(f"     {v:<30}{_stats(g.pts_full)}")
            eq = np.cumsum(F.sort_values("date").pts_full.values)
            L.append(f"   worst drawdown (whole position): {float((np.maximum.accumulate(eq) - eq).max()):.0f} pts")

    L += ["", "4) EACH MODEL'S VOTE  (sign of signal vs open->close; 50% = coin flip)"]
    for col in [c_ for c_ in D.columns if c_.startswith("sig_")]:
        s = D[col].dropna()
        s = s[s.abs() > 0.02]
        if len(s) >= 20:
            L.append(f"   {col[4:]:<12}{len(s):>6} days  right {100*(np.sign(s) == np.sign(oc[s.index])).mean():5.1f}%")
    L += ["", "Notes: walk-forward (no future data). No historical option chains, so call/put walls, gamma flip and max pain",
          "are missing from backtest levels. Past results do not guarantee future results. Not financial advice."]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: 2024-09-01, the start of the 1-minute data)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--discord", action="store_true", help="post the summary to DISCORD_WEBHOOK_URL when done")
    ap.add_argument("--fresh", action="store_true", help="ignore saved progress")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("quant").setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")
    end = datetime.strptime(a.end, "%Y-%m-%d").date() if a.end else (datetime.now(q.ET).date() - timedelta(days=1))
    start = datetime.strptime(a.start, "%Y-%m-%d").date() if a.start else datetime(2024, 9, 1).date()
    if end < start or (end - start).days < 5:
        raise SystemExit(f"Date range {start} -> {end} is too short. Leave 'end' empty to run up to yesterday.")
    cfg = q.Config()
    q.MODULES = [(fn, k) for fn, k in q.MODULES if k not in SKIP]
    q.RUN_ORDER = [k for k in q.RUN_ORDER if k not in SKIP]
    raw = download_daily(cfg, start, end)
    intraday = download_intraday(cfg)
    days = [d for d in sorted(set(raw.index.date)) if start <= d <= end
            and not pd.isna(raw.loc[raw.index.date == d, ("Open", cfg.equity)].iloc[0])]
    done_days, rows, trades = set(), [], []
    if os.path.exists(OUT_DAYS) and not a.fresh:
        rows = pd.read_csv(OUT_DAYS).to_dict("records")
        done_days = {r["date"] for r in rows}
        if os.path.exists(OUT_TRADES):
            trades = pd.read_csv(OUT_TRADES).to_dict("records")
        log.info("resuming: %d days already done", len(done_days))
    todo = [d for d in days if d.isoformat() not in done_days]
    log.info("%d trading days to run (%s -> %s), %d with intraday data", len(todo), start, end, sum(d in intraday for d in todo))
    t0 = time.time()
    for n, d in enumerate(todo, 1):
        try:
            row, tr = run_day(cfg, raw, intraday, d)
            rows.append(row)
            trades += tr
        except Exception as ex:
            log.warning("%s failed: %s", d, ex)
            continue
        if n % 5 == 0 or n == len(todo):
            pd.DataFrame(rows).sort_values("date").to_csv(OUT_DAYS, index=False)
            if trades:
                pd.DataFrame(trades).sort_values(["date", "rank"]).to_csv(OUT_TRADES, index=False)
            per = (time.time() - t0) / n
            log.info("%d/%d days  (%.1fs per day, ~%.0f min left)", n, len(todo), per, per * (len(todo) - n) / 60)
    if not rows:
        raise SystemExit("No trading days could be run - check the dates.")
    D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    T = pd.DataFrame(trades).sort_values(["date", "rank"]).reset_index(drop=True) if trades else pd.DataFrame()
    D.to_csv(OUT_DAYS, index=False)
    if not T.empty:
        T.to_csv(OUT_TRADES, index=False)
    txt = summarise(D, T)
    open(OUT_TXT, "w").write(txt)
    print(txt)
    if a.discord and cfg.webhook:
        try:
            q._post(cfg, {"username": "NY Open Quant Desk", "content": "**Backtest finished** - summary attached."},
                    files={"files[0]": (OUT_TXT, txt.encode("utf-8"), "text/plain")})
            log.info("summary posted to Discord")
        except Exception as ex:
            log.error("Discord post failed (%s) - results are still saved. Check the DISCORD_WEBHOOK_URL secret.", ex)


if __name__ == "__main__":
    main()
