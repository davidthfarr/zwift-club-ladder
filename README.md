# Club Ladder route picker

Pick the Zwift route that maximizes your team's chance of winning a 5 v 5 Club Ladder match
(places score 10, 9, … 1; most points wins).

## Use it

**In your browser, nothing to install:** https://davidthfarr.github.io/zwift-club-ladder/

The hosted page runs the same Python model inside your browser (via [Pyodide](https://pyodide.org)).
The first visit downloads about 10 MB; ranking every route takes a few seconds. Rider data you
add never leaves your device: it's saved in your browser's own storage. **Load example teams**
gives you made-up riders to try it with.

**Locally, faster:**

```
pip install numpy          # only dependency
python server.py           # then open http://localhost:8765
```

- **Riders:** drag in power CSVs (or type a rider in by hand); see every rider's curve, warnings, and team median curves.
- **Match:** pick your team and the opponent, filter by class / world / duration, and rank every route by P(win).
  Click a route for the expected finishing order and what decides it.
- **Routes:** browse all routes with class, estimated time for any field level, and key efforts.

The local server only listens on this computer (127.0.0.1). Its data lives in `pulls/`; removed files go to `pulls/_removed/`.

## Files

| File | What it does |
|---|---|
| `ladder_power.py` | Loads rider power CSVs (5s–20m, W and W/kg, vELO); flags suspicious numbers |
| `ladder_routes.py` | Route physics, key efforts, expected duration, class (sprinter / puncheur / hilly / climber) |
| `ladder_sim.py` | Monte Carlo match simulation per route: E[points], SD, P(win); ranks routes |
| `backtest.py` | Scores predictions against real results; fits noise and vELO weight |
| `index.html`, `ladder_api.py` | The web app: one page, plus the API it calls (in the browser or via the server) |
| `server.py` | Optional local server (standard library only) |
| `data/routes.json`, `data/segments.json` | 293 cycling routes and their climbs/sprints, from the MIT-licensed `zwift-data` package |
| `export_routes.js` | Refreshes the route data when Zwift adds routes (`npm install zwift-data && node export_routes.js`) |
| `examples/` | Column templates, and synthetic riders and results for trying everything out |

Requires Python 3.9+ and numpy (`pip install -r requirements.txt`). The command-line tools do the same things as the web app.

## Try it without real data

```
python examples/make_synthetic.py
python ladder_sim.py --power examples/synthetic_power.csv --us "Red Kites" --them "Green Finches" --max-min 60
python backtest.py --power examples/synthetic_power.csv --results examples/synthetic_results.csv --fit
```

The synthetic results are generated with known noise and vELO settings, so `--fit` should recover them
(it does: noise 0.06, sprint extra 0.04, vELO weight 0.4).

## Data and privacy

Real rider data - names, weights, power curves, vELO - is other people's information, read from
profiles that require sign-in. It stays in `pulls/` and `results/`, which are git-ignored. Only the
code, the route data and synthetic examples are in this repository. If you publish results, ask the
riders or anonymise them.

## Workflow

1. **Pull power.** Send Claude the zwiftracing.app rider URLs for both teams, labelled by team.
   It returns a dated CSV in the `examples/power_template.csv` layout. Keep the pulls in `pulls/`.
2. **Check the pull.** `python ladder_power.py "pulls/*.csv"` prints every rider's W/kg curve
   and warns about likely misreads.
3. **Rank routes.**
   ```
   python ladder_sim.py --power "pulls/*.csv" --us "My Team" --them "Their Team" --max-min 60
   ```
   Useful filters: `--world watopia`, `--class climber --class hilly`, `--route <slug>`,
   `--min-min / --max-min` (estimated minutes), `--laps 2`, `--velo-weight 0.4`
   (0 = power only), `--detail` (lineups and expected places), `--out picks.csv`.
4. **Test accuracy.** After each match, add its finishing order to `results/ladder_results.csv`
   (layout: `examples/results_template.csv`), then
   ```
   python backtest.py --power "pulls/*.csv" --results results/ladder_results.csv --fit
   ```
   It reports winner accuracy, Brier score, log loss, rank correlation, pairwise order accuracy
   and calibration, and fits the noise and vELO weight to the matches so far.
5. **Browse routes on their own.** `python ladder_routes.py --world london --detail`
   (`--ref-wkg 4.2` to time them for a faster field).

## How the model works

- **Key efforts.** Each route becomes a list of efforts that decide the finishing order:
  every named climb (timed with bike physics at this field's own power), the finish sprint
  (down-weighted after hard late climbs or on uphill finishes), unnamed rolling terrain, and
  the race-long pace. Climbs nearer the finish count more.
- **Body mass.** Each effort scores power as W / kg^e, with e = 1.0 on steep climbs (pure W/kg)
  falling to 0.67 on the flat, where absolute watts matter more.
- **Rolling terrain.** Climbing that isn't a named climb is modelled as repeated climbs whose size
  and gradient grow with the route's m/km (e.g. 5 x 60 m at 5%), timed with the same physics.
  It also thins the bunch, so it discounts the finish sprint.
- **Rider score.** The weighted average of log effective power across a route's efforts, read
  from the rider's power curve (log-log interpolation between the 7 durations).
- **vELO blend.** 40% of each rider's score (adjustable) comes from ZwiftRacing vELO, made
  route-specific: the sprint factor for the finish, punch for climbs of 2 min or less, climb for
  5 min or more (blended between), endurance or pursuit for race pace. 100 vELO2 points is
  scored as ~10% performance (fit on the first pull: r = 0.94 against power scores).
- **Match.** Score + Gaussian noise (more on sprint-decided routes) → ranks → 10…1 points.
  20,000 simulations per route. With more than 5 riders on a team, the best 5 for each route race.
- **Class rules.** climber: a climb of 10+ min, or 18+ m/km. hilly: a 5+ min
  climb or 10+ m/km. puncheur: 6+ m/km or a meaningful short climb. sprinter: everything else.

## Limitations and next steps

- **Noise levels are priors until fitted.** `backtest.py --fit` fits them (and the vELO weight)
  to real results. With fewer than ~15 matches, treat the fit as a hint.
- **Leakage in early backtests.** Power and vELO pulled after a match include that match, so
  backtests on past races look better than live predictions. Pull before each match from now on.
- **Durations are estimates, roughly ±10%.** They come from steady-state physics; there's no
  modelling of surges, attacks or Zwift's own draft model.
- **Route data gaps.** 70 routes have no named segments in `zwift-data`; their climbing is
  modelled as generic rolling climbs.
- **Ignored for now:** powerups, team tactics, and correlated form within a team.
