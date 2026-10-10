#!/usr/bin/env python3
"""
Full backtest of the NY Open Quant Desk bot.

For every trading day in [start, end] it rebuilds exactly what the bot would have known before
that day's 09:30 open (history up to the prior close + that day's official open), runs every model
that votes on the bias or sets the levels, and then checks the call against what really happened.

Skipped for speed because they have weight 0 (they never change the bias or levels):
Monte Carlo, finite differences, root-finding, lattices, RL execution.
Live-only differences: historical option chains aren't free, so implied vol comes from VIX (the bot's
own fallback) and SABR is skipped; 5-minute data only exists for ~60 days, so Hawkes and
Avellaneda-Stoikov only vote on recent dates.

    python backtest_full.py --start 2025-01-01 --end 2026-09-30
    python backtest_full.py --start 2025-01-01 --discord        # also post the summary to Discord
Results: backtest_days.csv (one row per day, resumable) and backtest_summary.txt
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import ny_open_quant as q

SKIP = {"mc", "fdm", "root", "lattice", "rl"}
OUT_CSV, OUT_TXT = "backtest_days.csv", "backtest_summary.txt"
log = logging.getLogger("backtest")


def download(cfg, start, end):
    tickers = list(dict.fromkeys([cfg.equity, cfg.pair, "IWM", "TLT", "GLD", "IWD", "IWF", "MTUM", "^VIX", "^IRX",
                                  "^FVX", "^TNX", "^TYX", cfg.futures]))
    s = (start - timedelta(days=int(365.25 * cfg.years) + 15)).isoformat()
    e = (end + timedelta(days=3)).isoformat()
    log.info("downloading daily history %s -> %s for %d tickers", s, e, len(tickers))
    raw = q.yf.download(tickers, start=s, end=e, interval="1d", auto_adjust=True, progress=False, group_by="column")
    intra = pd.DataFrame()
    try:
        intra = q._flat(q.yf.download(cfg.equity, period="60d", interval="5m", auto_adjust=True, progress=False))
        intra = intra.rename(columns=str.lower)
        if intra.index.tz is None:
            intra.index = intra.index.tz_localize("UTC")
        intra.index = intra.index.tz_convert(q.ET)
    except Exception as ex:
        log.warning("5-minute download failed (%s) - Hawkes/Avellaneda-Stoikov will be skipped", ex)
    return raw, intra


def context_for(cfg, raw, intra, day):
    """Exactly what load_live() builds, but as of `day`'s open, from pre-downloaded data."""
    now = datetime(day.year, day.month, day.day, 9, 30, 20, tzinfo=q.ET)
    lo = day - timedelta(days=int(365.25 * cfg.years))
    hist = raw[(raw.index.date < day) & (raw.index.date >= lo)]
    close = hist["Close"].copy().ffill()
    ohlc = pd.DataFrame({k.lower(): hist[k][cfg.equity] for k in ["Open", "High", "Low", "Close", "Volume"]}).dropna()
    px = close.reindex(ohlc.index).ffill()
    today = raw[raw.index.date == day]
    spot = float(today["Open"][cfg.equity].iloc[0])
    actual = {k: float(today[k.capitalize()][cfg.equity].iloc[0]) for k in ("open", "high", "low", "close")}
    it = pd.DataFrame()
    if len(intra):
        dts = intra.index.date
        it = intra[(dts < day) & (dts >= day - timedelta(days=45))]
    fut_ratio = float("nan")
    try:
        fut_ratio = float(px[cfg.futures].dropna().iloc[-1]) / float(px[cfg.equity].iloc[-1])
    except Exception:
        pass
    rf = float(px["^IRX"].dropna().iloc[-1]) / 100.0 if "^IRX" in px else 0.04
    c = q.Ctx(cfg=cfg, now=now, px=px, ohlc=ohlc, intra=it, spot=spot, fut_ratio=fut_ratio, rf=rf, q=cfg.div_yield)
    c.info["actual"] = actual
    return q.finish_ctx(c)


def run_day(cfg, raw, intra, day):
    c = context_for(cfg, raw, intra, day)
    res = q.run_all(c)
    k = q.compose(res, c)
    lv = {nm: v for nm, v, _ in k.lv}
    a = c.info["actual"]
    row = dict(date=day.isoformat(), verdict=k.verdict, score=k.score, p_up=k.p_up, agree=k.agree, move=k.move,
               open=a["open"], high=a["high"], low=a["low"], close=a["close"], prev_close=c.prev_close,
               fut_ratio=c.fut_ratio, XH=lv["Stretched upside"], UL=lv["Implied upside base"], centre=lv["Model centre"],
               DL=lv["Implied downside base"], XL=lv["Stretched downside"], pair_fair=c.info.get("pair_fair", np.nan))
    for r in res:
        row[f"sig_{r.key}"] = r.signal if r.ok else np.nan
    return row


# ----------------------------------------------------------------------------------------------
def summarise(D: pd.DataFrame) -> str:
    L = []
    O, H, Lo, C = D.open, D.high, D.low, D.close
    UL, DL, XH, XL = D.UL, D.DL, D.XH, D.XL
    oc = np.log(C / O)
    up = oc > 0
    n = len(D)
    L += ["=" * 92, f"NY OPEN QUANT DESK - FULL BACKTEST   {D.date.iloc[0]} -> {D.date.iloc[-1]}   {n} trading days", "=" * 92]

    L += ["", "1) THE RANGE  (what the report says vs what happened)",
          f"   {'claim':<60}{'claimed':>9}{'actual':>9}"]
    for lab, cl, act in [
        ("close between downside & upside limit", 68.3, ((C <= UL) & (C >= DL)).mean()),
        ("close inside extreme low / extreme high", 95.4, ((C <= XH) & (C >= XL)).mean()),
        ("'about 1 in 6 days close above' upside limit", 15.9, (C > UL).mean()),
        ("'about 1 in 6 days close below' downside limit", 15.9, (C < DL).mean()),
        ("close above extreme high", 2.3, (C > XH).mean()),
        ("close below extreme low", 2.3, (C < XL).mean()),
        ("price trades through upside limit during the day", None, (H >= UL).mean()),
        ("price trades through downside limit during the day", None, (Lo <= DL).mean()),
        ("session stays fully inside both limits", None, ((H < UL) & (Lo > DL)).mean()),
    ]:
        L.append(f"   {lab:<60}{(f'{cl:.1f}%' if cl else '-'):>9}{100*act:>8.1f}%")
    L.append(f"   real open->close spread / predicted move = {float(np.std(oc) / D.move.mean()):.2f}  (1.00 = perfectly sized)")

    L += ["", "2) THE BIAS  (did QQQ close above the open when the bot said UP, below when it said DOWN?)",
          f"   base rate: QQQ closed above its open on {100*up.mean():.1f}% of days"]
    for v in ("UP", "DOWN", "NEUTRAL / NO EDGE"):
        s = D.verdict == v
        if s.sum() == 0:
            L.append(f"   {v:<18} 0 days")
            continue
        r = oc[s]
        right = (r > 0).mean() if v == "UP" else (r < 0).mean() if v == "DOWN" else float("nan")
        L.append(f"   {v:<18} {s.sum():>4} days ({100*s.mean():4.1f}%)  closed up {100*(r>0).mean():5.1f}%  avg open->close {1e4*r.mean():+6.1f}bp"
                 + (f"  -> bias correct {100*right:.1f}%" if v != "NEUTRAL / NO EDGE" else ""))
    called = D.verdict != "NEUTRAL / NO EDGE"
    if called.any():
        hit = (np.sign(D.score[called]) == np.sign(oc[called])).mean()
        L.append(f"   overall: correct on {100*hit:.1f}% of days with a call (always guessing UP would be {100*up[called].mean():.1f}%)")
    L.append("   by strength of score (all days):")
    bins = [-1, -0.25, -0.12, 0, 0.12, 0.25, 1]
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        s = (D.score > lo_) & (D.score <= hi_)
        if s.sum():
            L.append(f"     score {lo_:+.2f}..{hi_:+.2f}: {s.sum():>4} days, closed up {100*up[s].mean():5.1f}%, avg {1e4*oc[s].mean():+6.1f}bp")
    L.append("   P(up) calibration (bot's stated odds vs reality):")
    for lo_, hi_ in ((0, .45), (.45, .49), (.49, .51), (.51, .55), (.55, 1)):
        s = (D.p_up > lo_) & (D.p_up <= hi_)
        if s.sum():
            L.append(f"     bot said {100*lo_:3.0f}-{100*hi_:3.0f}%: {s.sum():>4} days, actually up {100*up[s].mean():5.1f}%")

    L += ["", "3) EACH MODEL'S VOTE  (sign of signal vs open->close; 50% = coin flip)",
          f"   {'model':<12}{'days':>6}{'correct':>9}{'when +':>9}{'when -':>9}{'corr':>7}"]
    for col in [c for c in D.columns if c.startswith("sig_")]:
        s = D[col].dropna()
        s = s[s.abs() > 0.02]
        if len(s) < 20:
            continue
        r = oc[s.index]
        corr = float(np.corrcoef(s, r)[0, 1]) if s.std() > 0 else float("nan")
        L.append(f"   {col[4:]:<12}{len(s):>6}{100*(np.sign(s)==np.sign(r)).mean():>8.1f}%"
                 f"{100*(r[s>0]>0).mean() if (s>0).any() else float('nan'):>8.1f}%{100*(r[s<0]<0).mean() if (s<0).any() else float('nan'):>8.1f}%{corr:>7.3f}")

    L += ["", "4) THE PLAYBOOK  (daily bars: enter at the level, stop at the extreme, otherwise exit at the close;",
          "   NQ points, 1 contract, 0.75 pt cost. Daily bars can't show if a target came before or after the entry,",
          "   and the 5-min confirmation candle can't be checked, so entry is at the first touch of the level)"]
    rows = []
    for _, d in D.iterrows():
        ratio = d.fut_ratio if d.fut_ratio == d.fut_ratio else 41.0
        plays = []
        if d.verdict == "UP" and d.low <= d.DL:
            plays.append(("UP bias: buy downside limit", +1, d.DL, d.XL, d.UL))
        elif d.verdict == "DOWN" and d.high >= d.UL:
            plays.append(("DOWN bias: short upside limit", -1, d.UL, d.XH, d.DL))
        elif d.verdict == "NEUTRAL / NO EDGE":
            if d.high >= d.UL:
                plays.append(("No bias: short upside limit", -1, d.UL, d.XH, d.open))
            if d.low <= d.DL:
                plays.append(("No bias: buy downside limit", +1, d.DL, d.XL, d.open))
        for name, side, E, stop, tgt in plays:
            stopped = d.low <= stop if side > 0 else d.high >= stop       # the stop sits beyond the entry, so a stop touch always comes after entry
            pnl = side * (stop - E) if stopped else side * (d.close - E)
            rows.append(dict(play=name, pts=pnl * ratio - 0.75, risk=abs(E - stop) * ratio))
    T = pd.DataFrame(rows)
    if T.empty:
        L.append("   no playbook trades triggered")
    else:
        L.append(f"   {'play':<52}{'trades':>7}{'win':>8}{'avgRR':>7}{'pts/tr':>8}{'total':>8}{'MNQ $':>9}")
        for p, s in list(T.groupby("play")) + [("ALL PLAYBOOK TRADES", T)]:
            w, l = s.pts[s.pts > 0], s.pts[s.pts <= 0]
            rr = w.mean() / -l.mean() if len(w) and len(l) and l.mean() < 0 else float("nan")
            L.append(f"   {p:<52}{len(s):>7}{100*(s.pts>0).mean():>7.1f}%{rr:>7.2f}{s.pts.mean():>8.1f}{s.pts.sum():>8.0f}{2*s.pts.sum():>9,.0f}")

    L += ["", "5) BY YEAR", f"   {'year':<6}{'days':>5}{'inside limits':>15}{'bias calls':>12}{'bias correct':>14}"]
    D2 = D.assign(year=pd.to_datetime(D.date).dt.year)
    for y, g in D2.groupby("year"):
        gi = g.index
        inside = ((C[gi] <= UL[gi]) & (C[gi] >= DL[gi])).mean()
        cl_ = g.verdict != "NEUTRAL / NO EDGE"
        hit = (np.sign(g.score[cl_]) == np.sign(oc[gi][cl_])).mean() if cl_.any() else float("nan")
        L.append(f"   {y:<6}{len(g):>5}{100*inside:>14.1f}%{cl_.sum():>12}{100*hit:>13.1f}%")
    L += ["", "Notes: walk-forward (no future data). Implied vol = 1.2 x VIX; SABR skipped (no historical option chains);",
          "Hawkes / Avellaneda-Stoikov only vote on the last ~60 days. Past results do not guarantee future results."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: one year before --end)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--discord", action="store_true", help="post the summary to DISCORD_WEBHOOK_URL when done")
    ap.add_argument("--fresh", action="store_true", help="ignore any saved progress in backtest_days.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("quant").setLevel(logging.ERROR)
    end = datetime.strptime(a.end, "%Y-%m-%d").date() if a.end else (datetime.now(q.ET).date() - timedelta(days=1))
    start = datetime.strptime(a.start, "%Y-%m-%d").date() if a.start else end - timedelta(days=365)
    if end < start:
        raise SystemExit(f"End date {end} is before start date {start}. Leave 'end' empty to run up to yesterday.")
    if (end - start).days < 5:
        raise SystemExit(f"Only {(end - start).days + 1} calendar day(s) between {start} and {end} - that's too short to test. "
                         "Leave 'end' empty to run up to yesterday, or pick an end date at least a few weeks later.")
    cfg = q.Config()
    q.MODULES = [(fn, k) for fn, k in q.MODULES if k not in SKIP]
    q.RUN_ORDER = [k for k in q.RUN_ORDER if k not in SKIP]
    raw, intra = download(cfg, start, end)
    days = [d for d in sorted(set(raw.index.date)) if start <= d <= end
            and not pd.isna(raw.loc[raw.index.date == d, ("Open", cfg.equity)].iloc[0])]
    done = {}
    if os.path.exists(OUT_CSV) and not a.fresh:
        prev = pd.read_csv(OUT_CSV)
        done = {r["date"]: r for r in prev.to_dict("records")}
        log.info("resuming: %d days already done", len(done))
    rows, t0 = list(done.values()), time.time()
    todo = [d for d in days if d.isoformat() not in done]
    log.info("%d trading days to run (%s -> %s)", len(todo), start, end)
    for n, d in enumerate(todo, 1):
        try:
            rows.append(run_day(cfg, raw, intra, d))
        except Exception as ex:
            log.warning("%s failed: %s", d, ex)
            continue
        if n % 5 == 0 or n == len(todo):
            pd.DataFrame(rows).sort_values("date").to_csv(OUT_CSV, index=False)
            per = (time.time() - t0) / n
            log.info("%d/%d days  (%.1fs per day, ~%.0f min left)", n, len(todo), per, per * (len(todo) - n) / 60)
    if not rows:
        raise SystemExit(f"No trading days could be run between {start} and {end} (holidays/weekends only, or the "
                         "data download failed). Check the dates and try again.")
    D = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    D.to_csv(OUT_CSV, index=False)
    txt = summarise(D)
    open(OUT_TXT, "w").write(txt)
    print(txt)
    if a.discord and cfg.webhook:
        q._post(cfg, {"username": "NY Open Quant Desk", "content": "**Backtest finished** - summary attached."},
                files={"files[0]": (OUT_TXT, txt.encode("utf-8"), "text/plain")})
        log.info("summary posted to Discord")


if __name__ == "__main__":
    main()
