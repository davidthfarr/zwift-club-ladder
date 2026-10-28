#!/usr/bin/env python3
"""
make_synthetic.py - fake riders and fake Club Ladder results with KNOWN noise and
vELO settings, for testing backtest.py (and for the repo's example data, so no
real rider data needs to be published).

    python examples/make_synthetic.py            # writes examples/synthetic_*.csv
    python backtest.py --power examples/synthetic_power.csv \
                       --results examples/synthetic_results.csv --fit

If the harness works, --fit should land near TRUE below.
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import ladder_power as lp   # noqa: E402
import ladder_routes as lr  # noqa: E402
import ladder_sim as ls     # noqa: E402

TRUE = dict(base=0.06, sprint_extra=0.04, velo_weight=0.4)
TEAMS = ["Red Kites", "Blue Herons", "Green Finches", "Gold Owls", "Grey Wolves", "Teal Otters"]
RIDERS_PER_TEAM = 6
N_MATCHES = 60
SEED = 2026

# phenotype templates: W/kg relative to 20-min W/kg at 5s,15s,30s,1m,2m,5m,20m
PHENO = {
    "sprinter": [4.1, 3.4, 2.6, 1.85, 1.45, 1.18, 1.0],
    "puncheur": [3.4, 2.9, 2.35, 1.80, 1.48, 1.20, 1.0],
    "climber":  [2.9, 2.5, 2.10, 1.65, 1.40, 1.17, 1.0],
    "allround": [3.3, 2.8, 2.25, 1.72, 1.43, 1.18, 1.0],
}
FACTOR_OF = {"sprint": 0, "punch": 3, "climb": 5, "endurance": 6, "pursuit": 6, "tt": 6}


def make_riders(rng):
    riders, rid = [], 100001
    for t in TEAMS:
        for k in range(RIDERS_PER_TEAM):
            ph = rng.choice(list(PHENO))
            w20 = rng.normal(3.7, 0.35)
            kg = float(np.clip(rng.normal(76 if ph != "climber" else 68, 7), 55, 105))
            wkg = [w20 * m * rng.normal(1, 0.04) for m in PHENO[ph]]
            wkg = sorted(wkg, reverse=True)  # keep the curve monotone
            craft = rng.normal(0, 40)  # racecraft the power curve can't see, in vELO points
            factors = {f: 650 + (math.log(wkg[i] / PHENO["allround"][i] / 3.7) / 0.001) + craft + rng.normal(0, 15)
                       for f, i in FACTOR_OF.items()}
            velo = float(np.mean(list(factors.values())))
            riders.append(dict(rider_id=str(rid), name=f"{t.split()[1][:-1]} {k + 1}", team=t, weight_kg=round(kg, 1),
                               wkg=wkg, velo=velo, factors=factors))
            rid += 1
    return riders


def write_power(riders, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=lp.COLUMNS)
        w.writeheader()
        for r in riders:
            row = {c: "" for c in lp.COLUMNS}
            row.update(rider_id=r["rider_id"], name=r["name"], team=r["team"], weight_kg=r["weight_kg"],
                       velo=round(r["velo"], 1), pulled_at="2026-01-01")
            for f, v in r["factors"].items():
                row[f"velo_{f}"] = round(v)
            for (lbl, _), v in zip(lp.DURATIONS, r["wkg"]):
                row[f"wkg_{lbl}"] = round(v, 2)
                row[f"w_{lbl}"] = round(v * r["weight_kg"])
            w.writerow(row)


def main():
    rng = np.random.default_rng(SEED)
    riders = make_riders(rng)
    power_path = HERE / "synthetic_power.csv"
    write_power(riders, power_path)

    loaded = lp.load_riders(str(power_path))
    routes = [r for r in lr.load_routes().values() if not r.event_only]
    by_team = {t: [r for r in riders if r["team"] == t] for t in TEAMS}
    rows = []
    for m in range(N_MATCHES):
        ta, tb = rng.choice(TEAMS, 2, replace=False)
        lineup = list(rng.choice(by_team[ta], 5, replace=False)) + list(rng.choice(by_team[tb], 5, replace=False))
        while True:
            route = routes[rng.integers(len(routes))]
            ents = []
            for r in lineup:
                ents += ls.build_entrants({r["rider_id"]: loaded[r["rider_id"]]}, r["team"])
            model = lr.model_route(route, lr.median_curve([e.curve for e in ents]))
            if model.duration_s <= 75 * 60:
                break
        power = {id(e): lr.rider_score(e.curve, model.efforts) for e in ents}
        score = ls.blended_scores(ents, power, TRUE["velo_weight"], model.efforts, model.duration_s)
        sigma = math.hypot(TRUE["base"] + TRUE["sprint_extra"] * model.sprint_share, ls.NOISE["data"])
        perf = {id(e): score[id(e)] + rng.normal(0, sigma) for e in ents}
        order = sorted(ents, key=lambda e: -perf[id(e)])
        for pos, e in enumerate(order, 1):
            rows.append(dict(match_id=f"SYN-{m + 1:03d}", date=f"2026-02-{1 + m % 28:02d}", route_slug=route.slug,
                             laps=1, rider_id=e.rider.rider_id, team=e.team, finish_pos=pos))
    res_path = HERE / "synthetic_results.csv"
    with open(res_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(riders)} riders -> {power_path.name}; {N_MATCHES} matches -> {res_path.name}")
    print(f"true settings: {TRUE}")


if __name__ == "__main__":
    main()
