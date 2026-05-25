"""Survey: how often is ego physically on a sign-bound lane?

For each scene in MapDR Test split, compute the min lateral offset from
ego trajectory points to any lane referenced by label.json rule.centerline.
Used to decide demo scope (Option 2: filter subset; Option 1: redesign).

Output: experiments/trajectory_smoke/_filter_stats.json + console table.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# reuse project_point_to_polyline from smoke
from trajectory_demo_smoke import project_point_to_polyline, arc_lengths  # noqa: E402

MAPDR_ROOT = Path(os.environ.get("MAPDR_ROOT", str(REPO / "data" / "MapDR"))).resolve()
SPLIT_JSON = Path(os.environ.get("SPLIT_JSON", str(MAPDR_ROOT.parent / "split.json")))
OUT_PATH = REPO / "experiments/trajectory_smoke/_filter_stats.json"


def survey_scene(scene_dir: Path) -> dict | None:
    try:
        data = json.loads((scene_dir / "data.json").read_text())
        label = json.loads((scene_dir / "label.json").read_text())
    except Exception:
        return None
    poses = data["camera_pose"]
    ts = sorted(poses.keys(), key=int)
    if len(ts) < 2:
        return None
    ego = np.array([poses[t]["tvec_enu"] for t in ts], dtype=np.float64)

    # All vec_ids referenced by any rule.centerline (= sign-bound lanes)
    vectors = data["vector"]
    sign_lane_vids = set()
    for rule in label.values():
        for vid in rule.get("centerline", []):
            sign_lane_vids.add(str(vid))

    if not sign_lane_vids:
        return {
            "sid": scene_dir.name, "n_sign_lanes": 0,
            "lat_start_min": None, "lat_mean_min": None,
            "lat_full_min": None, "frac_on_lane_2m": None,
        }

    # For each sign-bound lane, compute lateral offset to ego start / mean / per-frame
    best_start = np.inf
    best_mean = np.inf
    best_full = -1.0   # min over lanes of (max over frames of lat_off)
    # frac_on_lane_2m: per ego frame, exists a sign-bound lane with lat_off < 2m?
    on_lane_per_frame = np.zeros(len(ego), dtype=bool)

    for vid in sign_lane_vids:
        if vid not in vectors:
            continue
        poly = np.asarray(vectors[vid]["vec_geo"], dtype=np.float64)
        if len(poly) < 2:
            continue
        arc = arc_lengths(poly)
        # lateral offsets ego_t -> this lane
        lats = np.array([project_point_to_polyline(p, poly, arc)[2] for p in ego])
        d_start = float(lats[0])
        d_mean = float(lats.mean())
        d_max = float(lats.max())
        if d_start < best_start:
            best_start = d_start
        if d_mean < best_mean:
            best_mean = d_mean
        # min over lanes of (max over frames): if some lane keeps ego close throughout
        if best_full < 0 or d_max < best_full:
            best_full = d_max
        on_lane_per_frame |= (lats < 2.0)

    frac = float(on_lane_per_frame.mean())
    return {
        "sid": scene_dir.name,
        "n_sign_lanes": len(sign_lane_vids),
        "lat_start_min": best_start,
        "lat_mean_min": best_mean,
        "lat_full_min": best_full,
        "frac_on_lane_2m": frac,
    }


def main():
    split = json.loads(SPLIT_JSON.read_text())
    test_ids = split["Test"]
    print(f"surveying {len(test_ids)} Test scenes...")

    stats = []
    for i, sid in enumerate(test_ids):
        sd = MAPDR_ROOT / sid
        if not sd.exists():
            continue
        s = survey_scene(sd)
        if s is None:
            continue
        stats.append(s)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(test_ids)} done")

    total = len(stats)
    print(f"\nsurvey done. {total} valid scenes.\n")

    # ------ thresholds table ------
    print(f"{'criterion':<55} {'count':>7} {'pct':>6}")
    print("-" * 70)
    for thresh in [1, 2, 3, 5]:
        n_start = sum(1 for s in stats
                      if s["lat_start_min"] is not None and s["lat_start_min"] < thresh)
        n_mean  = sum(1 for s in stats
                      if s["lat_mean_min"] is not None and s["lat_mean_min"] < thresh)
        n_full  = sum(1 for s in stats
                      if s["lat_full_min"] is not None and s["lat_full_min"] < thresh)
        print(f"ego_start lat_off < {thresh}m to any sign-bound lane          {n_start:>7} {n_start/total:>6.1%}")
        print(f"ego_mean  lat_off < {thresh}m to some sign-bound lane         {n_mean:>7} {n_mean/total:>6.1%}")
        print(f"ego_max-over-traj < {thresh}m to one sign-bound lane (strict) {n_full:>7} {n_full/total:>6.1%}")
        print()

    # ------ frac-on-lane distribution ------
    fracs = sorted(s["frac_on_lane_2m"] for s in stats if s["frac_on_lane_2m"] is not None)
    if fracs:
        print(f"frac of frames where ego is < 2m to any sign-bound lane (per scene):")
        for p in [10, 25, 50, 75, 90]:
            v = fracs[int(len(fracs) * p / 100)]
            print(f"  p{p:>2} = {v:.0%}")

        # Useful demo subset = ego on sign-bound lane >= 80% of frames
        n_80 = sum(1 for f in fracs if f >= 0.80)
        n_50 = sum(1 for f in fracs if f >= 0.50)
        print()
        print(f"scenes with ≥ 80% frames on a sign-bound lane: {n_80}/{total} ({n_80/total:.1%})  ← strict demo subset")
        print(f"scenes with ≥ 50% frames on a sign-bound lane: {n_50}/{total} ({n_50/total:.1%})  ← lenient demo subset")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"n_total": total, "per_scene": stats}, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
