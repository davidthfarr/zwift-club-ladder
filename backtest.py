#!/usr/bin/env python3
"""
backtest.py - how accurate is the model? Score its predictions against real
Club Ladder results, and fit the noise and vELO settings to them.

Results file (one row per rider per match; see examples/results_template.csv):
    match_id, date, route_slug, laps, rider_id, team, finish_pos
finish_pos is the rider's place among the match riders (1 = first). Leave it
blank for a DNF (scored last). Riders are looked up by rider_id in the power CSVs.

    python backtest.py --power "pulls/*.csv" --results results/ladder_results.csv
    python backtest.py ... --fit          # also fit noise + vELO weight to the results
    python backtest.py ... --out bt.csv   # per-match predictions

Reported
    winner accuracy   share of matches where the favoured team won
    Brier score       mean (P(win) - outcome)^2; 0.25 = coin flip, lower is better
    log loss          mean -log P(actual winner); 0.693 = coin flip, lower is better
    rank correlation  Spearman correlation of predicted vs actual finishing order
    pairwise accuracy share of rider pairs whose order the model got right
    calibration       predicted vs actual win rate in probability bins

Data leakage warning: power and vELO should be from BEFORE each match. Power
rows whose pulled_at is after the match date still get used if nothing older
exists, and those matches are flagged; current 90-day bests include the match
itself, so early backtests will look somewhat better than live predictions.

Fitting (--fit) maximises the pairwise likelihood of the observed finishing
orders, P(i ahead of j) = Phi((score_i - score_j) / (sigma * sqrt 2)), over a
grid of noise levels and vELO weights. With few matches the fit is noisy; the
output says how many matches and rider pairs it rests on.
"""

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

import ladder_power as lp
import ladder_routes as lr
import ladder_sim as ls

POINTS = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


@dataclass
class Match:
    match_id: str
    date: Optional[datetime]
    route_slug: str
    laps: int
    rows: List[dict]  # rider_id, team, finish_pos


def load_results(path: str) -> List[Match]:
    by_id: Dict[str, Match] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            mid = (row.get("match_id") or "").strip()
            if not mid:
                continue
            m = by_id.get(mid)
            if m is None:
                d = (row.get("date") or "").strip()
                m = by_id[mid] = Match(mid, datetime.strptime(d[:10], "%Y-%m-%d") if d else None,
                                       (row.get("route_slug") or "").strip(),
                                       int(row.get("laps") or 1), [])
            pos = (row.get("finish_pos") or "").strip()
            m.rows.append({"rider_id": (row.get("rider_id") or "").strip(),
                           "team": (row.get("team") or "").strip(),
                           "finish_pos": int(pos) if pos else None})
    return list(by_id.values())


def team_points(order_teams: List[str]) -> Dict[str, int]:
    pts: Dict[str, int] = defaultdict(int)
    for place, t in enumerate(order_teams):
        pts[t] += POINTS[place] if place < len(POINTS) else 0
    return pts


@dataclass
class Prepared:
    match: Match
    model: lr.RouteModel
    entrants: List[ls.Entrant]
    actual_order: List[int]   # indices into entrants, best first
    teams: List[str]
    winner: Optional[str]     # None = tie
    leaked: bool
    power: Dict[int, float]


def prepare(matches: List[Match], riders: Dict[str, lp.Rider], routes: Dict[int, Dict[str, lr.Route]]):
    out, skipped = [], []
    for m in matches:
        if m.laps not in routes:
            routes[m.laps] = lr.load_routes(m.laps)
        route = routes[m.laps].get(m.route_slug)
        if route is None:
            skipped.append(f"{m.match_id}: unknown route '{m.route_slug}'")
            continue
        ents, missing, leaked = [], [], False
        for row in m.rows:
            r = riders.get(row["rider_id"])
            if r is None:
                missing.append(row["rider_id"])
                continue
            if m.date and r.pulled_at and r.pulled_at.date() > m.date.date():
                leaked = True
            r2 = lp.Rider(**{**r.__dict__, "team": row["team"]})
            e = ls.build_entrants({r2.rider_id: r2}, row["team"])[0]
            ents.append((e, row["finish_pos"]))
        teams = sorted({e.team for e, _ in ents})
        if missing or len(teams) != 2:
            skipped.append(f"{m.match_id}: " + (f"no power data for {', '.join(missing)}" if missing
                                                 else f"needs exactly 2 teams, found {teams}"))
            continue
        big = 10 ** 6
        order = sorted(range(len(ents)), key=lambda i: ents[i][1] if ents[i][1] is not None else big)
        entrants = [e for e, _ in ents]
        pts = team_points([entrants[i].team for i in order])
        winner = None if pts[teams[0]] == pts[teams[1]] else max(teams, key=lambda t: pts[t])
        field = lr.median_curve([e.curve for e in entrants])
        model = lr.model_route(route, field)
        power = {id(e): lr.rider_score(e.curve, model.efforts) for e in entrants}
        out.append(Prepared(m, model, entrants, order, teams, winner, leaked, power))
    return out, skipped


def sigma_for(model: lr.RouteModel, base: float, sprint_extra: float) -> float:
    return math.hypot(base + sprint_extra * model.sprint_share, ls.NOISE["data"])


def predict(p: Prepared, base, sprint_extra, velo_w, sims, rng):
    """P(team A wins), P(tie), expected place per entrant - same simulation as ladder_sim."""
    score = ls.blended_scores(p.entrants, p.power, velo_w, p.model.efforts, p.model.duration_s)
    mu = np.array([score[id(e)] for e in p.entrants])
    k = len(mu)
    perf = mu + rng.normal(0.0, sigma_for(p.model, base, sprint_extra), size=(sims, k))
    order = np.argsort(-perf, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(sims)[:, None], order] = np.arange(k)
    pts_tab = np.array(POINTS + [0] * max(0, k - 10), dtype=float)[:k]
    pts = pts_tab[ranks]
    is_a = np.array([1.0 if e.team == p.teams[0] else 0.0 for e in p.entrants])
    margin = pts @ is_a - pts @ (1 - is_a)
    return float((margin > 0).mean()), float((margin == 0).mean()), ranks.mean(axis=0) + 1, mu


def pairwise_ll(p: Prepared, base, sprint_extra, velo_w) -> (float, int):
    score = ls.blended_scores(p.entrants, p.power, velo_w, p.model.efforts, p.model.duration_s)
    s = sigma_for(p.model, base, sprint_extra) * math.sqrt(2)
    ll, n = 0.0, 0
    order = p.actual_order
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            z = (score[id(p.entrants[i])] - score[id(p.entrants[j])]) / s
            ll += math.log(max(1e-12, 0.5 * math.erfc(-z / math.sqrt(2))))
            n += 1
    return ll, n


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate(prepared: List[Prepared], base, sprint_extra, velo_w, sims, seed, verbose=True):
    rng = np.random.default_rng(seed)
    rows, briers, lls, rhos, pair_ok, pair_n, correct = [], [], [], [], 0, 0, 0
    for p in prepared:
        p_a, p_tie, exp_place, mu = predict(p, base, sprint_extra, velo_w, sims, rng)
        actual_place = np.empty(len(p.entrants))
        actual_place[p.actual_order] = np.arange(1, len(p.entrants) + 1)
        rho = spearman(exp_place, actual_place)
        rhos.append(rho)
        for a in range(len(p.actual_order)):
            for b in range(a + 1, len(p.actual_order)):
                pair_n += 1
                pair_ok += mu[p.actual_order[a]] > mu[p.actual_order[b]]
        if p.winner is not None:
            y = 1.0 if p.winner == p.teams[0] else 0.0
            briers.append((p_a - y) ** 2)
            p_actual = p_a if y else 1 - p_a - p_tie
            lls.append(-math.log(max(1e-6, p_actual)))
            correct += (p_a > 0.5) == bool(y)
        rows.append({"match_id": p.match.match_id, "date": p.match.date.date().isoformat() if p.match.date else "",
                     "route": p.model.route.name, "class": p.model.route_class,
                     "team_a": p.teams[0], "team_b": p.teams[1], "p_a_wins": round(p_a, 3),
                     "p_tie": round(p_tie, 3), "winner": p.winner or "tie", "spearman": round(rho, 3),
                     "leakage": "yes" if p.leaked else ""})
    decided = len(briers)
    summary = {
        "matches": len(prepared), "decided": decided,
        "winner_acc": correct / decided if decided else float("nan"),
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("nan"),
        "spearman": float(np.nanmean(rhos)) if rhos else float("nan"),
        "pairwise_acc": pair_ok / pair_n if pair_n else float("nan"),
        "pairs": pair_n,
    }
    if verbose:
        _report(rows, summary)
    return rows, summary


def _report(rows, s):
    print(f"\n{'match':<14}{'route':<24}{'class':<10}{'A':<12}{'P(A)':>6}{'winner':>12}{'rho':>7}")
    for r in rows:
        flag = " *" if r["leakage"] else ""
        print(f"{r['match_id'][:13]:<14}{r['route'][:23]:<24}{r['class']:<10}{r['team_a'][:11]:<12}"
              f"{r['p_a_wins']:>6.0%}{r['winner'][:11]:>12}{r['spearman']:>7.2f}{flag}")
    if any(r["leakage"] for r in rows):
        print("  * power/vELO data pulled after this match (leakage)")
    print(f"\nMatches {s['matches']} ({s['decided']} decided)   rider pairs {s['pairs']}")
    print(f"Winner accuracy  {s['winner_acc']:.0%}    (favourite won)")
    print(f"Brier score      {s['brier']:.3f}  (coin flip 0.250)")
    print(f"Log loss         {s['log_loss']:.3f}  (coin flip 0.693)")
    print(f"Rank correlation {s['spearman']:.2f}   (Spearman, predicted vs actual order)")
    print(f"Pairwise order   {s['pairwise_acc']:.0%}    (rider pairs ordered correctly)")
    bins = defaultdict(list)
    for r in rows:
        if r["winner"] == "tie":
            continue
        p = r["p_a_wins"]
        fav_p, fav_won = (p, r["winner"] == r["team_a"]) if p >= 0.5 else (1 - p, r["winner"] == r["team_b"])
        bins[min(4, int((fav_p - 0.5) / 0.1))].append((fav_p, fav_won))
    if bins:
        print("\nCalibration (favourite's predicted vs actual win rate)")
        for b in sorted(bins):
            v = bins[b]
            print(f"  {50 + 10 * b:>3}-{60 + 10 * b:<3}%  n={len(v):<4} predicted {np.mean([x for x, _ in v]):.0%}"
                  f"  actual {np.mean([y for _, y in v]):.0%}")


def fit(prepared: List[Prepared]):
    bases = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    extras = [0.0, 0.04, 0.08, 0.12]
    vws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    best = None
    grid = []
    for vw in vws:
        for b in bases:
            for x in extras:
                tot, n = 0.0, 0
                for p in prepared:
                    ll, k = pairwise_ll(p, b, x, vw)
                    tot += ll
                    n += k
                grid.append((tot / max(n, 1), b, x, vw))
                if best is None or tot > best[0]:
                    best = (tot, b, x, vw, n)
    _, b, x, vw, n = best
    print(f"\nFit on {len(prepared)} matches / {n} rider pairs (pairwise likelihood):")
    print(f"  noise base {b}  sprint_extra {x}  vELO weight {vw}")
    print(f"  current     base {ls.NOISE['base']}  sprint_extra {ls.NOISE['sprint_extra']}  vELO weight {ls.VELO['weight']}")
    by_vw = {}
    for ll, bb, xx, v in grid:
        by_vw[v] = max(by_vw.get(v, -1e9), ll)
    print("  best mean log-lik by vELO weight: " + ", ".join(f"{v}: {by_vw[v]:.3f}" for v in vws))
    if len(prepared) < 15:
        print("  (fewer than 15 matches: treat the fitted values as a hint, not a setting)")
    return b, x, vw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", nargs="+", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--fit", action="store_true", help="fit noise + vELO weight, then re-evaluate with them")
    ap.add_argument("--velo-weight", type=float, default=ls.VELO["weight"])
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", help="write per-match predictions to CSV")
    a = ap.parse_args()

    riders = lp.load_riders(*a.power)
    prepared, skipped = prepare(load_results(a.results), riders, {})
    for s in skipped:
        print("skipped", s)
    if not prepared:
        raise SystemExit("no usable matches")

    print("== current settings ==")
    rows, _ = evaluate(prepared, ls.NOISE["base"], ls.NOISE["sprint_extra"], a.velo_weight, a.sims, a.seed)
    if a.fit:
        b, x, vw = fit(prepared)
        print("\n== fitted settings (in-sample: optimistic) ==")
        rows, _ = evaluate(prepared, b, x, vw, a.sims, a.seed)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} matches to {a.out}")


if __name__ == "__main__":
    main()
