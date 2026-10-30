#!/usr/bin/env python3
"""
server.py - run the Club Ladder route picker locally.

    python server.py            # then open http://localhost:8765
    python server.py --port 9000

Needs only Python 3.9+ and numpy. Binds to 127.0.0.1, so it is reachable only
from this computer. Power CSVs live in ./pulls (uploads and manual entries go there).

The same page also runs with no server at all (GitHub Pages): it then loads the
model into the browser with Pyodide. See README, "Hosted version".
"""

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import ladder_api as api

ROOT = Path(__file__).parent
api.set_pulls_dir(ROOT / "pulls")

# read-only files the page may fetch (example data for the "try example teams" button)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/examples/synthetic_power.csv": ("examples/synthetic_power.csv", "text/csv; charset=utf-8"),
    "/examples/power_template.csv": ("examples/power_template.csv", "text/csv; charset=utf-8"),
}


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
            if path in api.GET:
                return self._send(200, api.GET[path](self._query()))
            if path in STATIC_FILES:
                rel, ctype = STATIC_FILES[path]
                return self._send(200, (ROOT / rel).read_bytes(), ctype)
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(400, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path not in api.POST:
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 5_000_000:
                return self._send(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, api.POST[path](body))
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
