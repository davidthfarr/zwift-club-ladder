#!/usr/bin/env python3
"""
ladder_routes.py - route physics, demand profiles and classification for Zwift Club Ladder.

For each route it builds a list of *key efforts* (the climbs, the finish sprint,
rolling terrain and the race-long "engine" effort). Each effort has:
    duration  - seconds, from bike physics at the riders' own power
    mass_exp  - how much body weight matters (1.0 = pure W/kg on a climb,
                0.67 = flat/aero, where absolute watts matter more)
    weight    - how much it decides the finishing order
That list drives three things:
    * route class: sprinter / puncheur / hilly / climber
    * expected race duration for a given field
    * the rider performance score used by ladder_sim.py

Route data comes from data/routes.json + data/segments.json (exported from the
MIT-licensed zwift-data package by export_routes.js).

Command line:
    python ladder_routes.py                    # classify every route for a B-level field
    python ladder_routes.py --world london     # one world
    python ladder_routes.py --ref-wkg 4.2      # different reference field (20-min W/kg)
    python ladder_routes.py --out routes_classified.csv
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DATA = Path(__file__).parent / "data"
DURATIONS = [5, 15, 30, 60, 120, 300, 1200]
LABELS = {5: "5s", 15: "15s", 30: "30s", 60: "1m", 120: "2m", 300: "5m", 1200: "20m"}

# ---- tunable model constants (all in one place so they can be calibrated) ----
P = dict(
    g=9.81, rho=1.2, crr=0.004, bike_kg=8.0,
    cda=0.30,            # solo CdA (m^2) for a ~75 kg rider; scales with mass^0.67
    draft=0.67,          # CdA multiplier sitting in the pack on flat/descents
    descent_cap_kph=70.0,
    flat_frac=0.80,      # pack power on flats as a fraction of P(race duration)
    climb_frac=0.97,     # power on a decisive climb as a fraction of P(climb duration)
    min_climb_gain=12.0, # m; smaller climbs are ignored as selective efforts
    sprint_s=20.0,       # finish sprint duration
    sprint_weight=120.0, # weight of a bunch finish (in "metres of late climbing")
    late_bonus=0.6,      # climbs near the finish count up to (1+late_bonus)x more
    engine_weight=60.0,
    rolling_weight=0.8,  # weight per metre of climbing not covered by named climbs
    sprint_decay_m=400.0,# metres of late climbing that cut the sprint's weight by ~63%
)

# Reference power curve (W/kg), roughly a B / Ruby-level racer; scaled by --ref-wkg.
REF_CURVE_WKG = {5: 12.5, 15: 10.5, 30: 8.3, 60: 6.3, 120: 5.0, 300: 4.3, 1200: 3.6}
REF_MASS = 75.0


# ------------------------------------------------------------------ power curve
class Curve:
    """Power-duration curve: log-log interpolation between the 7 known points,
    power-law extrapolation past 20 min (from the 5->20 min slope)."""

    def __init__(self, watts: Dict[int, Optional[float]], mass_kg: float):
        pts = sorted((d, w) for d, w in watts.items() if w)
        if len(pts) < 2:
            raise ValueError("need at least two power points")
        self.d = [math.log(d) for d, _ in pts]
        self.w = [math.log(w) for _, w in pts]
        self.mass = mass_kg
        # slope used beyond the last point (fatigue), clamped to sane values
        self.tail = max(-0.08, min(-0.03, (self.w[-1] - self.w[-2]) / (self.d[-1] - self.d[-2])))

    @classmethod
    def from_wkg(cls, wkg: Dict[int, float], mass_kg: float):
        return cls({d: v * mass_kg for d, v in wkg.items() if v}, mass_kg)

    def watts(self, seconds: float) -> float:
        x = math.log(max(seconds, 1.0))
        d, w = self.d, self.w
        if x <= d[0]:
            return math.exp(w[0])
        if x >= d[-1]:
            return math.exp(w[-1] + self.tail * (x - d[-1]))
        for i in range(1, len(d)):
            if x <= d[i]:
                t = (x - d[i - 1]) / (d[i] - d[i - 1])
                return math.exp(w[i - 1] + t * (w[i] - w[i - 1]))
        return math.exp(w[-1])


def median_curve(curves: Sequence[Curve]) -> Curve:
    """Pointwise median W/kg curve of a field (sets race pace)."""
    import statistics
    mass = statistics.median(c.mass for c in curves)
    wkg = {d: statistics.median(c.watts(d) / c.mass for c in curves) for d in DURATIONS}
    return Curve.from_wkg(wkg, mass)


def ref_curve(wkg20: float = 3.6, mass: float = REF_MASS) -> Curve:
    k = wkg20 / REF_CURVE_WKG[1200]
    return Curve.from_wkg({d: v * k for d, v in REF_CURVE_WKG.items()}, mass)


# ---------------------------------------------------------------------- physics
def speed(power_w: float, mass_kg: float, grade: float, cda: float) -> float:
    """Steady-state speed (m/s) for power on a grade (fraction), by bisection."""
    m = mass_kg + P["bike_kg"]
    theta = math.atan(grade)

    def need(v):
        return (0.5 * P["rho"] * cda * v ** 3
                + P["crr"] * m * P["g"] * math.cos(theta) * v
                + m * P["g"] * math.sin(theta) * v)

    lo, hi = 0.3, 40.0
    if need(hi) < power_w:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if need(mid) < power_w else (lo, mid)
    v = (lo + hi) / 2
    if grade < 0:
        v = min(v, P["descent_cap_kph"] / 3.6)
    return v


def cda_for(mass_kg: float, drafting: bool) -> float:
    c = P["cda"] * (mass_kg / REF_MASS) ** 0.67
    return c * (P["draft"] if drafting else 1.0)


# ----------------------------------------------------------------------- routes
@dataclass
class Climb:
    name: str
    start_km: float
    length_km: float
    grade: float  # fraction

    @property
    def gain_m(self):
        return self.length_km * 1000 * self.grade


@dataclass
class Route:
    id: int
    slug: str
    name: str
    world: str
    distance_km: float
    elevation_m: float
    lead_in_km: float
    lead_in_elev_m: float
    event_only: bool
    climbs: List[Climb] = field(default_factory=list)
    laps: int = 1

    @property
    def race_km(self):
        return self.lead_in_km + self.distance_km * self.laps

    @property
    def race_elev(self):
        return self.lead_in_elev_m + self.elevation_m * self.laps

    @property
    def m_per_km(self):
        return self.race_elev / self.race_km if self.race_km else 0.0

    def race_climbs(self) -> List[Climb]:
        """Climbs over all laps, positioned from the start line (lead-in included)."""
        out = []
        for lap in range(self.laps):
            off = self.lead_in_km + lap * self.distance_km
            out += [Climb(c.name, c.start_km + off, c.length_km, c.grade) for c in self.climbs]
        return out


def load_routes(laps: int = 1) -> Dict[str, Route]:
    segs = json.loads((DATA / "segments.json").read_text())
    routes = {}
    for r in json.loads((DATA / "routes.json").read_text()):
        climbs = []
        for s in r["segments"]:
            meta = segs.get(s["slug"], {})
            if meta.get("type") != "climb":
                continue
            length = max(0.05, s["to_km"] - s["from_km"])
            inc = meta.get("avg_incline_pct")
            if inc is None and meta.get("elevation_m"):
                inc = meta["elevation_m"] / (length * 10)
            if inc is None:
                continue
            climbs.append(Climb(meta.get("name", s["slug"]), s["from_km"], length, inc / 100))
        routes[r["slug"]] = Route(r["id"], r["slug"], r["name"], r["world"], r["distance_km"],
                                  r["elevation_m"], r["lead_in_km"], r["lead_in_elev_m"],
                                  r["event_only"], sorted(climbs, key=lambda c: c.start_km), laps)
    return routes


# ---------------------------------------------------------------- key efforts
@dataclass
class Effort:
    kind: str        # climb | sprint | rolling | engine
    label: str
    duration_s: float
    mass_exp: float
    weight: float
    at_km: float = 0.0


@dataclass
class RouteModel:
    route: Route
    efforts: List[Effort]
    duration_s: float
    route_class: str
    demand: Dict[int, float]  # share of weight on each of the 7 durations

    @property
    def sprint_share(self):
        tot = sum(e.weight for e in self.efforts) or 1
        return sum(e.weight for e in self.efforts if e.kind == "sprint") / tot

    @property
    def longest_climb_s(self):
        return max((e.duration_s for e in self.efforts if e.kind == "climb"), default=0.0)


def _mass_exp(grade: float) -> float:
    return 0.67 + 0.33 * min(1.0, max(0.0, grade) / 0.05)


def _climb_time(c: Climb, curve: Curve) -> float:
    """Fixed point: time on climb depends on power, power depends on time."""
    t = 300.0
    for _ in range(12):
        p = curve.watts(t) * P["climb_frac"]
        v = speed(p, curve.mass, c.grade, cda_for(curve.mass, drafting=False))
        t_new = c.length_km * 1000 / v
        if abs(t_new - t) < 0.5:
            break
        t = t_new
    return t


def model_route(route: Route, field_curve: Curve) -> RouteModel:
    """Key efforts, duration and class for a route raced by a field like field_curve."""
    climbs = [c for c in route.race_climbs() if c.gain_m >= P["min_climb_gain"]]
    climb_km = sum(c.length_km for c in climbs)
    climb_gain = sum(c.gain_m for c in climbs)
    resid_gain = max(0.0, route.race_elev - climb_gain)  # climbing not in named climbs
    m = field_curve.mass

    # --- duration (fixed point on race time T) ---
    T = route.race_km / 40 * 3600
    climb_times = [_climb_time(c, field_curve) for c in climbs]
    for _ in range(10):
        p_flat = field_curve.watts(T) * P["flat_frac"]
        descent_km = min(route.race_km - climb_km, climb_km)
        flat_km = max(0.0, route.race_km - climb_km - descent_km)
        # rolling: residual climbing spread over half the flat distance, descended on the other half
        g_roll = resid_gain / (flat_km * 500) if flat_km > 0.5 else 0.0
        t_flat = (flat_km * 500 / speed(p_flat, m, g_roll, cda_for(m, True))
                  + flat_km * 500 / speed(p_flat, m, -g_roll, cda_for(m, True)))
        g_desc = -(climb_gain / (descent_km * 1000)) if descent_km > 0.05 else 0.0
        t_desc = descent_km * 1000 / speed(p_flat * 0.6, m, g_desc, cda_for(m, True)) if descent_km else 0.0
        T_new = sum(climb_times) + t_flat + t_desc
        if abs(T_new - T) < 5:
            T = T_new
            break
        T = T_new

    # --- key efforts ---
    efforts: List[Effort] = []
    finish_km = route.race_km
    uphill_finish = None
    for c, t in zip(climbs, climb_times):
        pos = (c.start_km + c.length_km) / finish_km
        w = c.gain_m * (1 + P["late_bonus"] * pos ** 2)
        if finish_km - (c.start_km + c.length_km) < 1.5:
            uphill_finish = c
            w *= 1.5
        efforts.append(Effort("climb", c.name, t, _mass_exp(c.grade), w, c.start_km))

    sprint_w = P["sprint_weight"]
    if uphill_finish is not None:
        sprint_w *= 0.3
    # hard late climbs split the bunch -> fewer riders contest a sprint
    late_gain = sum(c.gain_m for c in climbs if c.start_km > 0.5 * finish_km)
    # unnamed climbing thins the bunch too; about half of it falls in the second half
    sprint_w *= math.exp(-(late_gain + 0.5 * resid_gain) / P["sprint_decay_m"])
    efforts.append(Effort("sprint", "finish sprint", P["sprint_s"], 0.67, sprint_w, finish_km))

    if resid_gain > 20:
        # Climbing not covered by named climbs, modelled as repeated short climbs whose
        # size and steepness grow with the route's overall climbing density.
        mpk = route.m_per_km
        h = min(resid_gain, 90.0, max(20.0, 6.0 * mpk))  # metres per rolling climb
        g = min(0.08, max(0.03, 0.02 + 0.003 * mpk))  # gradient
        n = max(1, round(resid_gain / h))
        roll = Climb("rolling", finish_km / 2, h / g / 1000, g)
        t_roll = _climb_time(roll, field_curve)
        efforts.append(Effort("rolling", f"{n}x rolling {h:.0f} m @ {100 * g:.0f}%", t_roll,
                              _mass_exp(g), resid_gain * P["rolling_weight"] * (1 + P["late_bonus"] / 3),
                              finish_km / 2))

    efforts.append(Effort("engine", "race-long pace", min(T, 3600.0),
                          _mass_exp(route.m_per_km / 1000 * 3), P["engine_weight"]))

    demand = _demand(efforts)
    return RouteModel(route, efforts, T, _classify(route, efforts, demand), demand)


def _demand(efforts: List[Effort]) -> Dict[int, float]:
    """Spread effort weight onto the 7 standard durations (log-linear)."""
    out = {d: 0.0 for d in DURATIONS}
    logs = [math.log(d) for d in DURATIONS]
    for e in efforts:
        x = math.log(min(max(e.duration_s, 5), 1200))
        for i in range(1, len(DURATIONS)):
            if x <= logs[i]:
                t = (x - logs[i - 1]) / (logs[i] - logs[i - 1])
                out[DURATIONS[i - 1]] += e.weight * (1 - t)
                out[DURATIONS[i]] += e.weight * t
                break
    tot = sum(out.values()) or 1
    return {d: v / tot for d, v in out.items()}


def _classify(route: Route, efforts: List[Effort], demand: Dict[int, float]) -> str:
    """Rule-based class from the longest decisive climb and overall climbing density."""
    climbs = [e for e in efforts if e.kind == "climb"]
    longest = max((e.duration_s for e in climbs), default=0)
    mpk = route.m_per_km
    if longest >= 600 or mpk >= 18:
        return "climber"
    if longest >= 300 or mpk >= 10:
        return "hilly"
    if mpk >= 6 or any(e.duration_s >= 30 and e.weight >= 25 for e in climbs):
        return "puncheur"
    return "sprinter"


# ------------------------------------------------------------- rider scoring
def rider_score(curve: Curve, efforts: List[Effort]) -> float:
    """Weighted log 'effective power' over a route's key efforts (higher = better)."""
    tot = sum(e.weight for e in efforts) or 1
    return sum(e.weight * math.log(curve.watts(e.duration_s) / curve.mass ** e.mass_exp)
               for e in efforts) / tot


# ------------------------------------------------------------------------ CLI
def fmt_dur(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world")
    ap.add_argument("--route", action="append", help="route slug (repeatable)")
    ap.add_argument("--ref-wkg", type=float, default=3.6, help="reference field 20-min W/kg (default 3.6)")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--include-event-only", action="store_true")
    ap.add_argument("--detail", action="store_true", help="list each route's key efforts")
    ap.add_argument("--out", help="write CSV")
    a = ap.parse_args()

    curve = ref_curve(a.ref_wkg)
    rows = []
    for r in load_routes(a.laps).values():
        if a.world and r.world != a.world:
            continue
        if a.route and r.slug not in a.route:
            continue
        if r.event_only and not a.include_event_only and not a.route:
            continue
        rm = model_route(r, curve)
        rows.append(rm)
    rows.sort(key=lambda m: (m.route.world, m.duration_s))

    print(f"{'route':<34}{'world':<12}{'class':<10}{'km':>6}{'m':>6}{'m/km':>6}{'est time':>10}  demand 5s..20m (%)")
    for m in rows:
        r = m.route
        dem = " ".join(f"{round(100 * m.demand[d]):>3}" for d in DURATIONS)
        print(f"{r.name[:33]:<34}{r.world[:11]:<12}{m.route_class:<10}{r.race_km:>6.1f}{r.race_elev:>6.0f}"
              f"{r.m_per_km:>6.1f}{fmt_dur(m.duration_s):>10}  {dem}")
        if a.detail:
            for e in m.efforts:
                print(f"      {e.kind:<8}{e.label[:28]:<30}{fmt_dur(e.duration_s):>9}  w={e.weight:6.1f}  mass^{e.mass_exp:.2f}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slug", "name", "world", "class", "race_km", "race_elev_m", "m_per_km",
                        "est_duration_min", "longest_climb_min", "sprint_share"]
                       + [f"demand_{LABELS[d]}" for d in DURATIONS])
            for m in rows:
                r = m.route
                w.writerow([r.slug, r.name, r.world, m.route_class, round(r.race_km, 2), round(r.race_elev),
                            round(r.m_per_km, 1), round(m.duration_s / 60, 1), round(m.longest_climb_s / 60, 1),
                            round(m.sprint_share, 3)] + [round(m.demand[d], 3) for d in DURATIONS])
        print(f"\nWrote {len(rows)} routes to {a.out}")


if __name__ == "__main__":
    main()
