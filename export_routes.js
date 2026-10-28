// Refresh route data:  npm install zwift-data && node export_routes.js
// Writes data/routes.json and data/segments.json (zwift-data is MIT-licensed).
const fs = require("fs"), path = require("path");
const z = require(process.argv[2] || "zwift-data");
const routes = z.routes.filter(r => r.sports.includes("cycling")).map(r => ({
  id: r.id, slug: r.slug, name: r.name, world: r.world, distance_km: r.distance, elevation_m: r.elevation,
  lead_in_km: r.leadInDistance || 0, lead_in_elev_m: r.leadInElevation || 0, event_only: !!r.eventOnly,
  level_locked: !!r.levelLocked, supports_tt: !!r.supportsTT,
  segments: (r.segmentsOnRoute || []).map(s => ({ slug: s.segment, from_km: s.from, to_km: s.to })),
  strava_segment_id: r.stravaSegmentId || null, zwiftinsider_url: r.zwiftInsiderUrl || null,
}));
const segments = Object.fromEntries(z.segments.map(s => [s.slug, {
  name: s.name, type: s.type, world: s.world, distance_km: s.distance,
  avg_incline_pct: s.avgIncline ?? null, elevation_m: s.elevation ?? null,
}]));
const out = path.join(__dirname, "data");
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, "routes.json"), JSON.stringify(routes, null, 1));
fs.writeFileSync(path.join(out, "segments.json"), JSON.stringify(segments, null, 1));
console.log(`${routes.length} cycling routes, ${Object.keys(segments).length} segments`);
