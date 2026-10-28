#!/usr/bin/env python3
"""
ladder_power.py - load rider power profiles from CSV for the Club Ladder course picker.

The CSV is produced by the agent pull (one row per rider, read from the
rider's zwiftracing.app profile page). Columns:

    rider_id, name, team, category, velo, velo_sprint, velo_punch, velo_climb,
    velo_endurance, velo_pursuit, velo_tt, weight_kg,
    w_5s, w_15s, w_30s, w_1m, w_2m, w_5m, w_20m,
    wkg_5s, wkg_15s, wkg_30s, wkg_1m, wkg_2m, wkg_5m, wkg_20m,
    source_url, pulled_at

Blank cells are allowed (missing data). If a rider appears in several CSVs or
several rows, the row with the latest pulled_at wins.

Use as a module:
    from ladder_power import load_riders, team_profile
    riders = load_riders("pulls/*.csv")
    us   = team_profile(riders, team="My Team")
    them = team_profile(riders, team="Opponents")

Or from the command line to sanity-check a pull:
    python ladder_power.py pulls/week3.csv
"""

import csv
import glob
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional

DURATIONS = [("5s", 5), ("15s", 15), ("30s", 30), ("1m", 60), ("2m", 120), ("5m", 300), ("20m", 1200)]
VELO_FACTORS = ["sprint", "punch", "climb", "endurance", "pursuit", "tt"]
COLUMNS = (["rider_id", "name", "team", "category", "velo"] + [f"velo_{f}" for f in VELO_FACTORS] + ["weight_kg"]
           + [f"w_{l}" for l, _ in DURATIONS] + [f"wkg_{l}" for l, _ in DURATIONS]
           + ["source_url", "pulled_at"])

# Sanity bounds for flagging likely transcription errors (w/kg).
WKG_BOUNDS = {5: (5, 30), 15: (4, 25), 30: (3, 18), 60: (2.5, 13), 120: (2, 10), 300: (1.5, 8), 1200: (1.2, 7)}


@dataclass
class Rider:
    rider_id: str
    name: str = ""
    team: str = ""
    category: str = ""
    velo: Optional[float] = None                                     # ZwiftRacing vELO race rating
    velo_factors: Dict[str, Optional[float]] = field(default_factory=dict)  # sprint/punch/climb/...
    weight_kg: Optional[float] = None
    watts: Dict[int, Optional[float]] = field(default_factory=dict)  # seconds -> W
    wkg: Dict[int, Optional[float]] = field(default_factory=dict)    # seconds -> W/kg
    source_url: str = ""
    pulled_at: Optional[datetime] = None

    def wkg_at(self, secs: int) -> Optional[float]:
        """W/kg at a duration, deriving from watts/weight if only watts were recorded."""
        v = self.wkg.get(secs)
        if v is None and self.watts.get(secs) and self.weight_kg:
            v = self.watts[secs] / self.weight_kg
        return v

    def warnings(self) -> List[str]:
        out = []
        prev = None
        for label, s in DURATIONS:
            v = self.wkg_at(s)
            if v is None:
                continue
            lo, hi = WKG_BOUNDS[s]
            if not lo <= v <= hi:
                out.append(f"{label} {v:.2f} W/kg outside {lo}-{hi}")
            if prev is not None and v > prev[1] + 0.05:
                out.append(f"{label} ({v:.2f}) higher than {prev[0]} ({prev[1]:.2f})")
            prev = (label, v)
        return out


def _num(s) -> Optional[float]:
    s = (s or "").strip().replace(",", "")
    for suffix in ("w/kg", "wkg", "w", "kg"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].strip()
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _date(s) -> Optional[datetime]:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            pass
    return None


def _row_to_rider(row: dict) -> Rider:
    r = Rider(
        rider_id=(row.get("rider_id") or "").strip(),
        name=(row.get("name") or "").strip(),
        team=(row.get("team") or "").strip(),
        category=(row.get("category") or "").strip(),
        velo=_num(row.get("velo")),
        velo_factors={f: _num(row.get(f"velo_{f}")) for f in VELO_FACTORS},
        weight_kg=_num(row.get("weight_kg")),
        source_url=(row.get("source_url") or "").strip(),
        pulled_at=_date(row.get("pulled_at")),
    )
    for label, s in DURATIONS:
        r.watts[s] = _num(row.get(f"w_{label}"))
        r.wkg[s] = _num(row.get(f"wkg_{label}"))
    if r.weight_kg is None:  # back out weight if both W and W/kg were recorded
        for s in (1200, 300, 60):
            if r.watts.get(s) and r.wkg.get(s):
                r.weight_kg = round(r.watts[s] / r.wkg[s], 1)
                break
    return r


def load_riders(*paths: str) -> Dict[str, Rider]:
    """Load one or more CSVs (globs allowed). Latest pulled_at per rider wins."""
    files = sorted({f for p in paths for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"no CSV matched {paths}")
    riders: Dict[str, Rider] = {}
    for f in files:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                r = _row_to_rider(row)
                if not r.rider_id:
                    continue
                old = riders.get(r.rider_id)
                if old is None or (r.pulled_at or datetime.min) >= (old.pulled_at or datetime.min):
                    riders[r.rider_id] = r
    return riders


def team_profile(riders: Dict[str, Rider], team: Optional[str] = None,
                 ids: Optional[Iterable[str]] = None, stat: str = "median") -> Dict[int, Optional[float]]:
    """Collapse a set of riders into one W/kg-by-duration profile (median, mean, or max)."""
    pick = [r for r in riders.values()
            if (team is None or r.team == team) and (ids is None or r.rider_id in set(ids))]
    agg = {"median": statistics.median, "mean": statistics.mean, "max": max}[stat]
    out = {}
    for _, s in DURATIONS:
        vals = [v for v in (r.wkg_at(s) for r in pick) if v is not None]
        out[s] = round(agg(vals), 2) if vals else None
    return out


def write_template(path: str) -> None:
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerow(COLUMNS)


def _report(riders: Dict[str, Rider]) -> None:
    hdr = f"{'rider':<24}{'team':<16}" + "".join(f"{l:>7}" for l, _ in DURATIONS)
    print(hdr + "\n" + "-" * len(hdr))
    for r in sorted(riders.values(), key=lambda r: (r.team, r.name)):
        cells = "".join(f"{(r.wkg_at(s) or 0):>7.2f}" if r.wkg_at(s) else f"{'-':>7}" for _, s in DURATIONS)
        print(f"{r.name[:23]:<24}{r.team[:15]:<16}{cells}")
        for w in r.warnings():
            print(f"{'':<4}! {w}")
    teams = sorted({r.team for r in riders.values()})
    if len(teams) > 1:
        print("\nTeam medians (W/kg):")
        for t in teams:
            p = team_profile(riders, team=t)
            print(f"{t[:23]:<40}" + "".join(f"{(p[s] or 0):>7.2f}" for _, s in DURATIONS))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--template":
        write_template(sys.argv[2] if len(sys.argv) > 2 else "power_template.csv")
    else:
        _report(load_riders(*sys.argv[1:]))
