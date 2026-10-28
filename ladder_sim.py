#!/usr/bin/env python3
"""
ladder_sim.py - expected score, variance and win probability for a Zwift Club
Ladder match on each route, and the routes that maximize your win probability.

vELO blend
    Power curves say what a rider can produce; vELO (ZwiftRacing's race rating)
    says how that has turned into results. Each rider's route score is blended:
        score = (1 - w) * power_score + w * velo_score
    velo_score is route-specific: the rider's vELO factor ratings are weighted by
    the route's key efforts (sprint -> sprint factor, short climbs and rolling ->
    punch, long climbs -> climb, race-long pace -> endurance or pursuit), then put
    on the power score's log scale (100 vELO2 points ~ 10%), centred on the field.
    Riders with only an overall vELO use that; riders with none keep their power
    score. w = --velo-weight (default 0.4; 0 = power only).

Scoring (Club Ladder): 5 v 5, finishing places score 10, 9, ... 1; most points wins.

Model
    1. For each route, ladder_routes.model_route() turns the route into key
       efforts (climbs, finish sprint, rolling terrain, race-long pace), timed
       with physics at the pace of this particular field.
    2. Each rider gets a route score: weighted log of effective power
       (W / kg^e, e from 0.67 on flats to 1.0 on climbs) over those efforts,
       read from their 5s..20m power curve.
    3. Race outcome = score + noise. Noise is larger on sprint-decided routes
       (positioning, timing) and includes uncertainty in the power data.
    4. Monte Carlo: rank the 10 riders, award 10..1, sum by team.
       Output E[points], SD, P(win) per route; sort by P(win).
    If a team lists more than 5 riders, the 5 best for that route are raced.

Noise levels are priors, not fitted values. Calibrate them against past
Club Ladder results before trusting small differences between routes.

Usage
    python ladder_sim.py --power "pulls/*.csv" --us "My Team" --them "Their Team"
    python ladder_sim.py --power pulls/wk3.csv --us Us --them Them --world watopia --max-min 60
    python ladder_sim.py --power pulls/wk3.csv --us Us --them Them --route tempus-fugit --detail
    python ladder_sim.py ... --class climber --class hilly --top 10 --out picks.csv
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import ladder_power as lp
import ladder_routes as lr

POINTS = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)

NOISE = dict(
    base=0.05,          # race-day form / tactics, in log-performance units (~5%)
    sprint_extra=0.08,  # added in proportion to how sprint-decided the route is
    data=0.02,          # staleness / error in the pulled power numbers
    missing_mass=75.0,  # kg assumed when a rider has no weight
)

VELO = dict(
    weight=0.4,            # share of the score that comes from vELO
    log_per_point=0.0010,  # vELO2 points -> log performance (fit on the first pull: 0.0009, r = 0.94)
)


def _factor_mix(e: lr.Effort, race_s: float) -> Dict[str, float]:
    """Which vELO factors an effort calls on. Climbs blend smoothly from punch
    (2 min and under) to climb (5 min and over)."""
    if e.kind == "sprint":
        return {"sprint": 1.0}
    if e.kind in ("climb", "rolling"):
        c = min(1.0, max(0.0, (e.duration_s - 120) / 180))
        return {"punch": 1 - c, "climb": c}
    return {"endurance": 1.0} if race_s >= 2400 else {"pursuit": 1.0}


def route_velo(r: lp.Rider, efforts: List[lr.Effort], race_s: float) -> Optional[float]:
    """Rider's vELO weighted by the factors this route's efforts call on."""
    num = den = 0.0
    for e in efforts:
        for f, share in _factor_mix(e, race_s).items():
            v = (r.velo_factors or {}).get(f)
            if v is None:
                v = r.velo
            if v is None or share == 0:
                continue
            num += e.weight * share * v
            den += e.weight * share
    return num / den if den else None


def blended_scores(entrants, power: Dict[int, float], weight: float,
                   efforts: List[lr.Effort], race_s: float) -> Dict[int, float]:
    """Blend each rider's route power score with a route-specific vELO score on the same scale."""
    vel = {id(e): route_velo(e.rider, efforts, race_s) for e in entrants}
    rated = [e for e in entrants if vel[id(e)] is not None]
    if weight <= 0 or len(rated) < 2:
        return dict(power)
    mean_p = sum(power[id(e)] for e in rated) / len(rated)
    mean_v = sum(vel[id(e)] for e in rated) / len(rated)
    out = dict(power)
    for e in rated:
        v_score = mean_p + VELO["log_per_point"] * (vel[id(e)] - mean_v)
        out[id(e)] = (1 - weight) * power[id(e)] + weight * v_score
    return out


@dataclass
class Entrant:
    rider: lp.Rider
    curve: lr.Curve
    team: str


def build_entrants(riders: Dict[str, lp.Rider], team: str) -> List[Entrant]:
    out = []
    for r in riders.values():
        if r.team != team:
            continue
        mass = r.weight_kg or NOISE["missing_mass"]
        watts = {s: (r.watts.get(s) or (r.wkg_at(s) * mass if r.wkg_at(s) else None)) for _, s in lp.DURATIONS}
        try:
            out.append(Entrant(r, lr.Curve(watts, mass), team))
        except ValueError:
            print(f"  skipping {r.name or r.rider_id}: not enough power data")
    if not out:
        raise SystemExit(f"no usable riders for team '{team}'")
    return out


@dataclass
class MatchResult:
    model: lr.RouteModel
    us_lineup: List[Entrant]
    them_lineup: List[Entrant]
    exp_us: float
    exp_them: float
    sd: float
    p_win: float
    p_tie: float
    exp_place: Dict[str, float]


def simulate(model_route_fn, route: lr.Route, us: List[Entrant], them: List[Entrant],
             n: int, rng: np.random.Generator, velo_weight: Optional[float] = None) -> MatchResult:
    field = lr.median_curve([e.curve for e in us + them])
    m = model_route_fn(route, field)
    power = {id(e): lr.rider_score(e.curve, m.efforts) for e in us + them}
    score = blended_scores(us + them, power, VELO["weight"] if velo_weight is None else velo_weight,
                           m.efforts, m.duration_s)
    us5 = sorted(us, key=lambda e: -score[id(e)])[:5]
    them5 = sorted(them, key=lambda e: -score[id(e)])[:5]
    racers = us5 + them5
    k = len(racers)
    mu = np.array([score[id(e)] for e in racers])
    sigma = math.hypot(NOISE["base"] + NOISE["sprint_extra"] * m.sprint_share, NOISE["data"])
    perf = mu + rng.normal(0.0, sigma, size=(n, k))
    order = np.argsort(-perf, axis=1)                  # best first
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(k)
    pts = POINTS[:k][ranks] if k <= 10 else np.where(ranks < 10, POINTS[np.minimum(ranks, 9)], 0)
    is_us = np.array([1.0] * len(us5) + [0.0] * len(them5))
    us_pts = pts @ is_us
    them_pts = pts @ (1 - is_us)
    margin = us_pts - them_pts
    exp_place = {e.rider.name or e.rider.rider_id: float(ranks[:, i].mean() + 1) for i, e in enumerate(racers)}
    return MatchResult(m, us5, them5, float(us_pts.mean()), float(them_pts.mean()), float(us_pts.std()),
                       float((margin > 0).mean()), float((margin == 0).mean()), exp_place)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", nargs="+", required=True, help="power CSV(s) from the pulls (globs ok)")
    ap.add_argument("--us", required=True, help="our team label in the CSV 'team' column")
    ap.add_argument("--them", required=True, help="opponent team label")
    ap.add_argument("--route", action="append", help="only these route slugs")
    ap.add_argument("--world", action="append", help="only these worlds")
    ap.add_argument("--class", dest="classes", action="append",
                    choices=["sprinter", "puncheur", "hilly", "climber"], help="only these classes")
    ap.add_argument("--min-min", type=float, default=0, help="min est. duration (minutes)")
    ap.add_argument("--max-min", type=float, default=1e9, help="max est. duration (minutes)")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--include-event-only", action="store_true")
    ap.add_argument("--velo-weight", type=float, default=VELO["weight"],
                    help="share of each rider's score from vELO, 0-1 (default 0.4; 0 = power only)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--detail", action="store_true", help="show lineups and expected places")
    ap.add_argument("--out", help="write all results to CSV")
    a = ap.parse_args()

    riders = lp.load_riders(*a.power)
    us, them = build_entrants(riders, a.us), build_entrants(riders, a.them)
    rng = np.random.default_rng(a.seed)

    results = []
    for r in lr.load_routes(a.laps).values():
        if a.route and r.slug not in a.route:
            continue
        if a.world and r.world not in a.world:
            continue
        if r.event_only and not a.include_event_only and not a.route:
            continue
        res = simulate(lr.model_route, r, us, them, a.sims, rng, a.velo_weight)
        mins = res.model.duration_s / 60
        if not (a.min_min <= mins <= a.max_min):
            continue
        if a.classes and res.model.route_class not in a.classes:
            continue
        results.append(res)
    if not results:
        raise SystemExit("no routes matched the filters")
    results.sort(key=lambda x: -x.p_win)

    print(f"\n{a.us} ({len(us)} riders) vs {a.them} ({len(them)} riders) - {a.sims:,} sims per route\n")
    print(f"{'route':<32}{'world':<11}{'class':<10}{'time':>8}{'E[us]':>7}{'E[them]':>8}{'SD':>5}{'P(win)':>8}")
    for x in results[:a.top]:
        r = x.model.route
        print(f"{r.name[:31]:<32}{r.world[:10]:<11}{x.model.route_class:<10}{lr.fmt_dur(x.model.duration_s):>8}"
              f"{x.exp_us:>7.1f}{x.exp_them:>8.1f}{x.sd:>5.1f}{x.p_win:>7.0%}")
        if a.detail:
            for name, place in sorted(x.exp_place.items(), key=lambda kv: kv[1]):
                side = a.us if any((e.rider.name or e.rider.rider_id) == name for e in x.us_lineup) else a.them
                print(f"      {place:4.1f}  {name[:28]:<30}{side}")

    best, worst = results[0], results[-1]
    print(f"\nBest: {best.model.route.name} ({best.p_win:.0%})   Worst: {worst.model.route.name} ({worst.p_win:.0%})")
    by_class: Dict[str, List[float]] = {}
    for x in results:
        by_class.setdefault(x.model.route_class, []).append(x.p_win)
    print("Mean P(win) by class: " + ", ".join(f"{c} {np.mean(v):.0%}" for c, v in sorted(by_class.items())))

    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slug", "name", "world", "class", "est_duration_min", "exp_points_us", "exp_points_them",
                        "sd_points_us", "p_win", "p_tie", "us_lineup", "them_lineup"])
            for x in results:
                r = x.model.route
                w.writerow([r.slug, r.name, r.world, x.model.route_class, round(x.model.duration_s / 60, 1),
                            round(x.exp_us, 2), round(x.exp_them, 2), round(x.sd, 2), round(x.p_win, 4),
                            round(x.p_tie, 4), "; ".join(e.rider.name for e in x.us_lineup),
                            "; ".join(e.rider.name for e in x.them_lineup)])
        print(f"Wrote {len(results)} routes to {a.out}")


if __name__ == "__main__":
    main()
