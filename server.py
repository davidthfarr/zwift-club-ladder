#!/usr/bin/env python3
"""
server.py - local web app for the Club Ladder route picker.

    python server.py            # then open http://localhost:8765
    python server.py --port 9000

Needs only Python 3.9+ and numpy. Binds to 127.0.0.1, so it is reachable only
from this computer. Power CSVs live in ./pulls (uploads and manual entries go there).
"""

import argparse
import csv
import io
import json
import re
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import ladder_power as lp
import ladder_routes as lr
import ladder_sim as ls

ROOT = Path(__file__).parent
PULLS = ROOT / "pulls"
STATIC = ROOT / "static"
PULLS.mkdir(exist_ok=True)
MANUAL = PULLS / "manual_entries.csv"


# ------------------------------------------------------------------ helpers
def load_all_riders():
    files = list(PULLS.glob("*.csv"))
    return lp.load_riders(*[str(f) for f in files]) if files else {}


def rider_json(r: lp.Rider):
    return {
        "rider_id": r.rider_id, "name": r.name, "team": r.team, "category": r.category, "velo": r.velo,
        "weight_kg": r.weight_kg, "source_url": r.source_url,
        "pulled_at": r.pulled_at.date().isoformat() if r.pulled_at else None,
        "wkg": {lbl: (round(r.wkg_at(s), 2) if r.wkg_at(s) else None) for lbl, s in lp.DURATIONS},
        "watts": {lbl: (round(r.watts[s]) if r.watts.get(s) else
                        (round(r.wkg_at(s) * r.weight_kg) if r.wkg_at(s) and r.weight_kg else None))
                  for lbl, s in lp.DURATIONS},
        "warnings": r.warnings(),
    }


def model_json(m: lr.RouteModel):
    r = m.route
    return {
        "slug": r.slug, "name": r.name, "world": r.world, "class": m.route_class,
        "km": round(r.race_km, 1), "elev_m": round(r.race_elev), "m_per_km": round(r.m_per_km, 1),
        "duration_s": round(m.duration_s), "longest_climb_s": round(m.longest_climb_s),
        "sprint_share": round(m.sprint_share, 3), "event_only": r.event_only,
        "demand": {lr.LABELS[d]: round(v, 3) for d, v in m.demand.items()},
        "efforts": [{"kind": e.kind, "label": e.label, "duration_s": round(e.duration_s),
                     "weight": round(e.weight, 1), "mass_exp": round(e.mass_exp, 2),
                     "at_km": round(e.at_km, 1)} for e in m.efforts],
    }


def route_filter(r: lr.Route, q: dict):
    worlds = q.get("worlds") or []
    if worlds and r.world not in worlds:
        return False
    slugs = q.get("routes") or []
    if slugs and r.slug not in slugs:
        return False
    if r.event_only and not q.get("include_event_only") and not slugs:
        return False
    return True


def post_filter(m: lr.RouteModel, q: dict):
    mins = m.duration_s / 60
    if mins < float(q.get("min_min") or 0) or mins > float(q.get("max_min") or 1e9):
        return False
    classes = q.get("classes") or []
    return not classes or m.route_class in classes


# --------------------------------------------------------------------- API
def api_riders(_q):
    riders = load_all_riders()
    teams = sorted({r.team for r in riders.values() if r.team})
    return {"riders": [rider_json(r) for r in sorted(riders.values(), key=lambda r: (r.team, r.name))],
            "teams": teams, "files": sorted(f.name for f in PULLS.glob("*.csv"))}


def api_routes(q):
    curve = lr.ref_curve(float(q.get("ref_wkg") or 3.6))
    out = []
    for r in lr.load_routes(int(q.get("laps") or 1)).values():
        if not route_filter(r, q):
            continue
        m = lr.model_route(r, curve)
        if post_filter(m, q):
            out.append(model_json(m))
    worlds = sorted({r.world for r in lr.load_routes().values()})
    return {"routes": sorted(out, key=lambda x: (x["world"], x["duration_s"])), "worlds": worlds}


def api_simulate(q):
    riders = load_all_riders()
    us = ls.build_entrants(riders, q["us"])
    them = ls.build_entrants(riders, q["them"])
    sims = max(1000, min(50000, int(q.get("sims") or 10000)))
    rng = np.random.default_rng(int(q.get("seed") or 7))
    results = []
    for r in lr.load_routes(int(q.get("laps") or 1)).values():
        if not route_filter(r, q):
            continue
        vw = q.get("velo_weight")
        res = ls.simulate(lr.model_route, r, us, them, sims, rng,
                          None if vw in (None, "") else max(0.0, min(1.0, float(vw))))
        if not post_filter(res.model, q):
            continue
        us_names = {e.rider.name or e.rider.rider_id for e in res.us_lineup}
        d = model_json(res.model)
        d.update({
            "exp_us": round(res.exp_us, 2), "exp_them": round(res.exp_them, 2), "sd": round(res.sd, 2),
            "p_win": round(res.p_win, 4), "p_tie": round(res.p_tie, 4),
            "places": [{"name": n, "team": q["us"] if n in us_names else q["them"], "place": round(p, 2)}
                       for n, p in sorted(res.exp_place.items(), key=lambda kv: kv[1])],
        })
        results.append(d)
    results.sort(key=lambda x: -x["p_win"])
    by_class = {}
    for x in results:
        by_class.setdefault(x["class"], []).append(x["p_win"])
    return {"results": results, "sims": sims,
            "by_class": {c: round(float(np.mean(v)), 3) for c, v in sorted(by_class.items())},
            "us_n": len(us), "them_n": len(them)}


def api_upload(body):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", body.get("filename") or "upload.csv")
    if not name.endswith(".csv"):
        name += ".csv"
    text = body.get("content") or ""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "rider_id" not in rows[0]:
        raise ValueError("CSV needs a header row with at least a rider_id column (see examples/power_template.csv)")
    team_override = (body.get("team") or "").strip()
    if team_override:
        for r in rows:
            r["team"] = team_override
    if not any(r.get("pulled_at") for r in rows):
        for r in rows:
            r["pulled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    path = PULLS / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lp.COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"saved": name, "rows": len(rows)}


def api_add_rider(body):
    row = {c: "" for c in lp.COLUMNS}
    for k in ("rider_id", "name", "team", "category", "velo", "weight_kg", "source_url"):
        row[k] = str(body.get(k) or "").strip()
    if not row["rider_id"] or not row["team"]:
        raise ValueError("rider_id and team are required")
    for lbl, _ in lp.DURATIONS:
        row[f"w_{lbl}"] = str((body.get("watts") or {}).get(lbl) or "")
        row[f"wkg_{lbl}"] = str((body.get("wkg") or {}).get(lbl) or "")
    row["pulled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    new = not MANUAL.exists()
    with open(MANUAL, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lp.COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return {"saved": row["rider_id"]}


def api_delete_file(body):
    name = Path(body.get("filename") or "").name
    path = PULLS / name
    if not name.endswith(".csv") or not path.exists():
        raise ValueError("no such file")
    trash = PULLS / "_removed"
    trash.mkdir(exist_ok=True)
    path.rename(trash / f"{datetime.now():%Y%m%d-%H%M%S}-{name}")
    return {"removed": name}


GET = {"/api/riders": api_riders, "/api/routes": api_routes}
POST = {"/api/simulate": api_simulate, "/api/upload": api_upload,
        "/api/rider": api_add_rider, "/api/remove-file": api_delete_file}


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _query(self):
        q = {}
        for k, v in parse_qs(urlparse(self.path).query).items():
            q[k] = v if k in ("worlds", "classes", "routes") else v[0]
        return q

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in GET:
                return self._send(200, GET[path](self._query()))
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(400, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path not in POST:
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 5_000_000:
                return self._send(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, POST[path](body))
        except SystemExit as e:  # build_entrants signals "no riders for team"
            self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            print(f"{self.log_date_time_string()}  {args[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Club Ladder route picker: http://localhost:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
