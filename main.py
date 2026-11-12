#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: trajectory forecaster, no model calls.

Reads /input/task.json and /input/history.ndjson (10-minute market-cap
snapshots of pump.fun coins launched since T0) and writes
/output/forecast.json: for each of the 5 places, a probability for each
candidate coin; the top 5 in order; and for each place a distribution over
the 82 log10 market-cap bins.

Model, in brief: every coin's full trajectory is scored for persistence to
the answer reads 7-13 hours after the cut. The score blends the coin's
current rank and size, its momentum over the last 1h and 3h (median step,
so a single 10-minute spike does not dominate), its drawdown from its own
peak, how often it held a top rank, and whether it has graduated to a pool
(complete). Per-place probabilities spread over the ranked candidates with
a width that grows for lower places; the coins left unnamed are the live
mid-rank coins, while the deadest coins (crashed, old, never near the top
lately) are named at p 0 so they take nothing from the unnamed share.
Sizes are lognormals centred a little below the cut cap of the coin
expected at that place, with a wide sd for the 7-13h horizon: caps drift
down more often than up after the launch phase, and a too-narrow size
forecast is punished hard when it misses.

Deterministic: a pure function of the input files. Python 3.13 stdlib only.
"""
import json
import math
import os
from collections import defaultdict

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
FLOOR = 1e-4


def parse_time(s):
    # ISO 8601 UTC, e.g. 2026-11-11T20:10:06.480Z -> epoch seconds (approx).
    import datetime
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def logcap(lamports):
    # log10 of the market cap in SOL.
    return math.log10(float(lamports)) - 9.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = task["size"]["edges"]
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    marks.sort(key=lambda m: m["at"])
    cut_t = parse_time(marks[-1]["at"])

    # Per-coin trajectory: list of (time, rank, logcap, venue, complete).
    series = defaultdict(list)
    created = {}
    for m in marks:
        t = parse_time(m["at"])
        for r in m["rows"]:
            mint = r["mint"]
            series[mint].append(
                (t, int(r["rank"]), logcap(r["market_cap_lamports"]),
                 r.get("venue"), bool(r.get("complete")))
            )
            if mint not in created and r.get("created_at"):
                created[mint] = parse_time(r["created_at"])

    last = marks[-1]["rows"]
    last_chart = sorted(last, key=lambda r: r["rank"])
    last_rank = {r["mint"]: int(r["rank"]) for r in last}
    last_cap = {r["mint"]: logcap(r["market_cap_lamports"]) for r in last}

    def window_points(mint, hours):
        t0 = cut_t - hours * 3600
        return [p for p in series[mint] if p[0] >= t0]

    def feat(mint):
        pts = series[mint]
        t_now, rank_now, cap_now, venue_now, complete_now = pts[-1]
        caps = [p[2] for p in pts]
        peak = max(caps)
        # Momentum: median 10-minute log step over the last hour, and the
        # 3-hour log change; a single spike cannot dominate either.
        def log_change(hours):
            w = window_points(mint, hours)
            if len(w) < 2:
                return 0.0
            return w[-1][2] - w[0][2]
        steps = [pts[i + 1][2] - pts[i][2] for i in range(len(pts) - 1)]
        recent_steps = [s for (t, *_), s in zip(pts[:-1], steps) if t >= cut_t - 3600]
        med_step = sorted(recent_steps)[len(recent_steps) // 2] if recent_steps else 0.0
        # Rank persistence over the last 3 hours.
        w3 = window_points(mint, 3)
        top10_share = (sum(1 for p in w3 if p[1] <= 10) / len(w3)) if w3 else 0.0
        top20_share = (sum(1 for p in w3 if p[1] <= 20) / len(w3)) if w3 else 0.0
        mean_rank3 = (sum(p[1] for p in w3) / len(w3)) if w3 else 99.0
        age_h = (cut_t - created[mint]) / 3600.0 if mint in created else 99.0
        return {
            "rank": rank_now, "cap": cap_now, "peak": peak,
            "drawdown": peak - cap_now,
            "chg1h": log_change(1), "chg3h": log_change(3),
            "med_step": med_step,
            "top10": top10_share, "top20": top20_share,
            "mean_rank3": mean_rank3,
            "graduated": 1.0 if (complete_now or venue_now == "pool") else 0.0,
            "age_h": age_h, "n": len(pts),
        }

    # Candidates: the last chart's coins, plus any coin that was top-30 in
    # the last 3 hours (a recent faller can climb back).
    cand = set(last_rank)
    for m in marks:
        if parse_time(m["at"]) >= cut_t - 3 * 3600:
            for r in m["rows"]:
                if int(r["rank"]) <= 30:
                    cand.add(r["mint"])
    feats = {mint: feat(mint) for mint in cand}

    def score(mint):
        f = feats[mint]
        s = 0.0
        # Current standing dominates: rank and size at the cut.
        s += -0.55 * math.log(max(f["rank"], 1))
        s += 0.9 * (f["cap"] - 3.0)
        # Momentum: sustained growth helps, a crash hurts; cap the terms so
        # one wild coin cannot run away with the ranking.
        s += 1.1 * max(-0.6, min(0.6, f["chg1h"]))
        s += 0.7 * max(-0.8, min(0.8, f["chg3h"]))
        s += 2.0 * max(-0.15, min(0.15, f["med_step"]))
        # Drawdown from the coin's own peak: a coin that already gave back
        # most of its pump rarely returns to the top.
        s += -0.9 * min(f["drawdown"], 1.5)
        # Persistence and graduation: holding a top rank and being on a pool
        # is what survives the next 7-13 hours.
        s += 0.8 * f["top10"] + 0.4 * f["top20"]
        s += -0.25 * math.log(max(f["mean_rank3"], 1))
        s += 0.35 * f["graduated"]
        # Very young coins are lottery tickets: strong if already top-5,
        # otherwise discounted; very old coins that never graduated fade.
        if f["age_h"] < 1.0 and f["rank"] > 10:
            s -= 0.5
        if f["age_h"] > 8.0 and not f["graduated"]:
            s -= 0.4
        return s

    scored = sorted(cand, key=lambda m: (-score(m), last_rank.get(m, 999)))
    expected_rank = {mint: i + 1 for i, mint in enumerate(scored)}
    top5 = scored[:places]

    # Per-place probabilities: a Gaussian bump around each coin's expected
    # rank, widening for lower places, times a quality weight from its
    # score gap to the best. Normalised per place to leave a small
    # unnamed share for surprise coins.
    qualities = {}
    best = score(scored[0])
    for mint in scored:
        qualities[mint] = math.exp(max(-4.0, min(0.0, (score(mint) - best) * 1.2)))
    probabilities = []
    for k in range(1, places + 1):
        sigma = 2.0 + 0.6 * k
        raw = {}
        for mint in scored:
            r = expected_rank[mint]
            w = qualities[mint] * math.exp(-((r - k) ** 2) / (2 * sigma * sigma))
            raw[mint] = w
        total = sum(raw.values())
        keep = {m: w / total for m, w in raw.items() if w / total >= 0.0015}
        # Leave ~8% unnamed for coins not named (incl. born after the cut).
        scale = 0.92 / sum(keep.values()) if keep else 0.0
        keep = {m: p * scale for m, p in keep.items()}
        # Temper the single biggest probability: no coin holds a place
        # with more than ~45% chance 7-13 hours out; spread the excess.
        for _ in range(3):
            big = [m for m in keep if keep[m] > 0.45]
            if not big:
                break
            for m in big:
                excess = keep[m] - 0.45
                keep[m] = 0.45
                others = [x for x in keep if x != m and keep[x] < 0.45]
                tot = sum(keep[x] for x in others)
                if tot > 0:
                    for x in others:
                        keep[x] += excess * keep[x] / tot
        for mint in sorted(keep, key=lambda m: -keep[m]):
            p = math.floor(keep[mint] * 1e6) / 1e6
            if p > 0:
                probabilities.append({"mint": mint, "place": k, "p": p})
        # p 0 for the deadest coins: crashed to under a tenth of peak,
        # old, far down the chart, no recent top-30. They take nothing
        # from the unnamed share if they stay dead, as they almost surely do.
        dead = []
        for mint in last_chart:
            mint = mint["mint"]
            f = feats.get(mint)
            if not f:
                continue
            if (f["rank"] > 60 and f["drawdown"] > 1.0 and f["age_h"] > 4.0
                    and f["top10"] == 0.0):
                dead.append(mint)
        for mint in dead[:30]:
            probabilities.append({"mint": mint, "place": k, "p": 0})

    # Sizes: lognormal per place. The centre and width are measured from
    # the history itself: the recent drift and volatility of the cap at
    # that rank over the last 3 hours, projected to the ~10h horizon
    # (the middle of the 7-13h answer window) and damped, because top
    # caps mean-revert instead of random-walking. A fixed wide sd would
    # throw probability away; a fixed narrow one is punished hard when
    # the cap moves.
    sizes = []
    import datetime as _dt
    mark_times = [parse_time(m["at"]) for m in marks]
    for k in range(1, places + 1):
        # Series of the log cap at rank k across marks.
        series_k = []
        for m in marks:
            rows = sorted(m["rows"], key=lambda r: r["rank"])
            if len(rows) >= k:
                series_k.append((parse_time(m["at"]),
                                 logcap(rows[k - 1]["market_cap_lamports"])))
        # Hourly log changes over the last 3 hours.
        t3 = cut_t - 3 * 3600
        pts3 = [(t, c) for (t, c) in series_k if t >= t3]
        hourly = []
        for i in range(1, len(pts3)):
            dt_h = (pts3[i][0] - pts3[i - 1][0]) / 3600.0
            if dt_h > 0:
                hourly.append((pts3[i][1] - pts3[i - 1][1]) / dt_h)
        if hourly:
            drift = sum(hourly) / len(hourly)
            # Robust volatility: median absolute deviation, so a launch
            # phase's accelerating trend does not inflate the width.
            med = sorted(hourly)[len(hourly) // 2]
            mad = sorted(abs(h - med) for h in hourly)[len(hourly) // 2]
            vol_h = 1.4826 * mad
        else:
            drift, vol_h = 0.0, 0.15
        # Damp the drift: only a third of the recent hourly drift is
        # projected, and the total shift is capped.
        shift = max(-0.40, min(0.20, drift * 10.0 / 3.0))
        mean = series_k[-1][1] + shift if series_k else feats[scored[k - 1]]["cap"]
        sd = max(0.35, min(0.55, vol_h * 2.2))
        for b in range(len(edges) + 1):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            p = math.floor(1e6 * (norm_cdf(hi, mean, sd) - norm_cdf(lo, mean, sd))) / 1e6
            if p > 0:
                sizes.append({"place": k, "bin": b, "p": p})

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps(out) + "\n")
    print(f"forecast: {len(probabilities)} probability rows, "
          f"{len(sizes)} size rows, top5={top5}")


def norm_cdf(x, mean, sd):
    if x == -math.inf:
        return 0.0
    if x == math.inf:
        return 1.0
    return 0.5 * math.erfc((mean - x) / (sd * math.sqrt(2.0)))


if __name__ == "__main__":
    main()
