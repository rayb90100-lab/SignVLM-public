"""Per-frame planner inference + BEV/dashcam render, side-by-side with geo baseline.

Output structure: experiments/trajectory_smoke/<scene_8>/planner/{bev,dashcam,metrics.json,overview_*.png}
— mirrors `geo/`, `no_perturb/`, `C05/` directory layout from trajectory_demo_smoke.py.

For each frame i in the scene:
  1. Build planner input (8-frame history ending at i, target_lane, scene_lanes, rule)
  2. Run planner → 20-frame future trajectory in ego frame
  3. Convert back to ENU world coords
  4. Sample to match baseline horizon (n_future=4 in 2s window) → fair ADE/FDE comparison

Usage:
    python scripts/trajectory_planner_render.py \
        --scene 9b7a2bcbb6904f77aed8fe4637553669 \
        --ckpt runs/planner/v1_100ep_n20/best.pt \
        --tag planner \
        --out-root experiments/trajectory_smoke

    # Or batch all 8 demo scenes:
    python scripts/trajectory_planner_render.py --all-demo
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Reuse load_scene + plot functions from existing trajectory_demo_smoke
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import src.projection` works
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_demo_smoke import (
    load_scene, gt_trajectory, all_centerlines,
    plot_bev_closedloop, plot_bev_mosaic, plot_bev_per_frame,
    plot_dashcam_per_frame, plot_perframe_metrics,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from planner_model import PlannerMLP

REPO = Path(__file__).resolve().parent.parent
MAPDR_ROOT = Path(os.environ.get('MAPDR_ROOT', str(REPO / 'data' / 'MapDR')))

# Match planner_extract_data.py constants
N_HISTORY = 8
N_FUTURE = 20
N_TARGET_LANE_PTS = 20
N_SCENE_LANES = 6
SPEED_NORM = 120.0
DIR_VOCAB = ['TurnLeft', 'TurnRight', 'GoStraight', 'UTurn']
VEH_VOCAB = ['Bus', 'Truck', 'NonMotor', 'Any']

M_EGO_FROM_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def build_ego_transform(t_ego_world: np.ndarray, q_xyzw: list):
    R_cam2world = R.from_quat(q_xyzw).as_matrix()
    R_world2ego = M_EGO_FROM_CAM @ R_cam2world.T
    return R_world2ego, R_world2ego.T  # world→ego, ego→world


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray | None:
    """Resample (N, D) polyline to n points uniformly along arclength."""
    if pts.shape[0] < 2:
        return None
    seg = np.diff(pts[:, :2], axis=0)
    cumlen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    if cumlen[-1] < 1e-6:
        return None
    t = np.linspace(0.0, cumlen[-1], n)
    out = np.stack([np.interp(t, cumlen, pts[:, k]) for k in range(pts.shape[1])], axis=1)
    return out


def build_rule_vector(attr_info: dict) -> np.ndarray:
    rule = np.zeros(10, dtype=np.float32)
    sl = str(attr_info.get('HighSpeedLimit', 'None'))
    if sl in ('None', 'N/A', 'null'):
        sl = str(attr_info.get('LowSpeedLimit', 'None'))
    try:
        rule[0] = float(sl) / SPEED_NORM
    except (ValueError, TypeError):
        rule[0] = -1.0
    dir_field = attr_info.get('LaneDirection', [])
    if isinstance(dir_field, str):
        dir_field = [dir_field]
    for d in dir_field:
        if d in DIR_VOCAB:
            rule[1 + DIR_VOCAB.index(d)] = 1.0
    eff_date = str(attr_info.get('EffectiveDate', 'None'))
    rule[5] = 0.0 if eff_date in ('None', 'N/A', '') else 1.0
    veh_field = attr_info.get('AllowedTransport', 'None')
    if isinstance(veh_field, str):
        veh_field = [veh_field]
    for v in veh_field:
        if v in VEH_VOCAB:
            rule[6 + VEH_VOCAB.index(v)] = 1.0
    if rule[6:10].sum() == 0:
        rule[6 + VEH_VOCAB.index('Any')] = 1.0
    return rule


def parse_attr_from_signvlm_jsonl(jsonl_path: str, scene_uid: str,
                                   prefer_frame: int = 0) -> dict | None:
    """Parse first rule's attr_info from a SignVLM inference jsonl pred_text.

    Tries `prefer_frame` first; on parse/empty-rules failure walks other frames.
    Returns None if every frame fails — caller falls back to GT label.
    """
    if not jsonl_path or not Path(jsonl_path).exists():
        return None
    frames = {}
    for line in open(jsonl_path):
        r = json.loads(line)
        if r.get("scene_id") != scene_uid:
            continue
        frames[r["frame_idx"]] = r.get("pred_text", "")
    if not frames:
        return None
    order = [prefer_frame] + [k for k in sorted(frames) if k != prefer_frame]
    for fi in order:
        text = frames.get(fi)
        if not text:
            continue
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines)
        d = None
        try:
            d = json.loads(s.replace("'", '"'))
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if m:
                try:
                    d = json.loads(m.group(0).replace("'", '"'))
                except json.JSONDecodeError:
                    pass
        if not d:
            continue
        rules = d.get("rules", []) or []
        if not rules:
            continue
        attr = rules[0].get("attr_info", {}) or {}
        if attr:
            return attr
    return None


def planner_short_window(scene: dict, model, device: str,
                         horizon_s: float = 2.0, n_future: int = 4,
                         signvlm_lanes: dict | None = None,
                         signvlm_attr: dict | None = None) -> dict:
    """Mirror baseline_short_window schema, but use learned planner for prediction."""
    ts_all, gt_all = gt_trajectory(scene["poses"])
    centerlines = all_centerlines(scene["vectors"])
    if not centerlines:
        return {"per_frame": [], "mean_ADE": None, "mean_FDE": None}
    vectors_by_vid = {vid: poly for vid, poly in centerlines}

    # Target lane vid: SignVLM-given (if provided), else from label.json first rule
    label = scene.get("label", {})
    label_target_vid = None
    rule_attr = {}
    if label:
        first_key = sorted(label.keys())[0]
        rule_attr = label[first_key].get('attr_info', {})
        cl_list = label[first_key].get('centerline', [])
        if cl_list:
            label_target_vid = str(cl_list[0])
    # If SignVLM attr_info provided (from jsonl pred_text), override GT rule.
    # This wires panel-perturb → SignVLM output → planner input, otherwise the
    # rule path silently bypasses SignVLM and reads GT directly.
    rule_attr_used = signvlm_attr if signvlm_attr else rule_attr
    rule_vec = build_rule_vector(rule_attr_used)  # (10,)
    rule_t = torch.from_numpy(rule_vec).unsqueeze(0).to(device)
    if signvlm_attr is not None:
        print(f'  [rule-from-signvlm] attr_info = {rule_attr_used}')
        print(f'  [rule-from-signvlm] rule_vec = {rule_vec.round(3).tolist()}')

    per_frame = []

    for i in range(len(ts_all)):
        t_i = ts_all[i]
        ego_xyz_i = gt_all[i]
        # Future window for ADE/FDE evaluation (same logic as geo baseline)
        future_ts = [t for t in ts_all[i+1:] if (t - t_i) <= horizon_s * 1e9]
        if len(future_ts) < 1:
            continue
        n = min(n_future, len(future_ts))
        idxs = np.linspace(0, len(future_ts) - 1, num=n).astype(int)
        sampled_ts = [future_ts[j] for j in idxs]
        gt_future = np.array([scene["poses"][str(t)]["tvec_enu"]
                              for t in sampled_ts], dtype=np.float64)
        rel_ts = np.array([(t - t_i) / 1e9 for t in sampled_ts])

        # === Build planner input (anchored at frame i) ===
        # Need 8 history frames ending at i (use repeat-pad if not enough)
        hist_idxs = [max(0, i - (N_HISTORY - 1 - k)) for k in range(N_HISTORY)]
        hist_ts = [ts_all[k] for k in hist_idxs]
        # Pose at anchor frame i
        pose_i = scene["poses"][str(t_i)]
        t0_tvec = np.array(pose_i['tvec_enu'], dtype=np.float64)
        R_w2e_i, R_e2w_i = build_ego_transform(t0_tvec, pose_i['rvec_enu'])

        # ego history (world→ego)
        hist_xy = np.zeros((N_HISTORY, 2))
        hist_yaw = np.zeros(N_HISTORY)
        for k, ts_k in enumerate(hist_ts):
            pose_k = scene["poses"][str(ts_k)]
            tvec_k = np.array(pose_k['tvec_enu'])
            xyz_ego = R_w2e_i @ (tvec_k - t0_tvec)
            hist_xy[k] = xyz_ego[:2]
            R_k = R.from_quat(pose_k['rvec_enu']).as_matrix()
            R_k_in_ego = R_w2e_i @ R_k
            hist_yaw[k] = np.arctan2(R_k_in_ego[1, 0], R_k_in_ego[0, 0])
        # velocity (finite diff)
        ts_ns = np.array([int(t) for t in hist_ts])
        dt = np.diff(ts_ns).astype(np.float64) / 1e9
        hist_v = np.zeros(N_HISTORY)
        if (dt > 0).any():
            v = np.linalg.norm(np.diff(hist_xy, axis=0), axis=1) / np.maximum(dt, 1e-3)
            hist_v[1:] = v
            hist_v[0] = v[0] if len(v) else 0.0
        ego_history = np.concatenate([hist_xy, hist_yaw[:, None], hist_v[:, None]], axis=1).astype(np.float32)

        # Determine target_lane_vid for this frame
        if signvlm_lanes is not None and i in signvlm_lanes:
            target_vid = signvlm_lanes[i]
        else:
            target_vid = label_target_vid
        if target_vid is None or str(target_vid) not in vectors_by_vid:
            per_frame.append({
                "frame_idx": i, "t_ns": t_i,
                "lane_vid": target_vid, "lateral_off_m": None, "s_along_lane": None,
                "pred": [], "gt_future": gt_future.tolist(),
                "ADE": None, "FDE": None, "skipped": True,
            })
            continue
        target_poly_world = vectors_by_vid[str(target_vid)]  # (N, 3) world
        target_poly_ego = (target_poly_world - t0_tvec) @ R_w2e_i.T  # (N, 3) ego
        target_lane = resample_polyline(target_poly_ego[:, :2], N_TARGET_LANE_PTS)
        if target_lane is None:
            continue

        # scene_lanes (other centerlines, padded)
        other_lanes = []
        for vid, poly_w in centerlines:
            if vid == str(target_vid):
                continue
            poly_e = (poly_w - t0_tvec) @ R_w2e_i.T
            rs = resample_polyline(poly_e[:, :2], N_TARGET_LANE_PTS)
            if rs is not None:
                other_lanes.append(rs)
        other_lanes.sort(key=lambda cl: np.linalg.norm(cl[N_TARGET_LANE_PTS // 2]))
        scene_lanes = np.zeros((N_SCENE_LANES, N_TARGET_LANE_PTS, 2), dtype=np.float32)
        scene_mask = np.zeros(N_SCENE_LANES, dtype=np.float32)
        for k, cl in enumerate(other_lanes[:N_SCENE_LANES]):
            scene_lanes[k] = cl
            scene_mask[k] = 1.0

        # === Run planner ===
        batch = {
            'ego_history': torch.from_numpy(ego_history).unsqueeze(0).to(device),
            'target_lane': torch.from_numpy(target_lane.astype(np.float32)).unsqueeze(0).to(device),
            'scene_lanes': torch.from_numpy(scene_lanes).unsqueeze(0).to(device),
            'scene_mask': torch.from_numpy(scene_mask).unsqueeze(0).to(device),
            'rule': rule_t,
        }
        with torch.no_grad():
            pred_ego = model(batch).cpu().numpy()[0]  # (20, 2) ego frame

        # Convert pred (ego frame) → world ENU
        pred_xy_world = (R_e2w_i @ np.concatenate([pred_ego, np.zeros((N_FUTURE, 1))], axis=1).T).T + t0_tvec
        # Now sample to match rel_ts. Planner output is at scene frame rate; assume uniform across N_FUTURE.
        # Find each rel_t's nearest planner frame index.
        # Estimate scene frame rate from history dt
        dt_avg = float(np.median(dt)) if (dt > 0).any() else 0.25
        planner_rel_ts = np.arange(1, N_FUTURE + 1) * dt_avg
        pred_xyz_at_rel_ts = np.stack([
            np.interp(rel_ts, planner_rel_ts, pred_xy_world[:, 0]),
            np.interp(rel_ts, planner_rel_ts, pred_xy_world[:, 1]),
            np.interp(rel_ts, planner_rel_ts, pred_xy_world[:, 2]),
        ], axis=1)
        # Planner predicts 2D (x, y) in ego frame; the z reconstructed above assumes
        # ego_z=0 which is "level relative to camera" — for a pitched-down dashcam that
        # plane rises above the road, so pred projects above gt in the dashcam image.
        # Pin pred ENU z to gt's actual camera height so the dashcam overlay aligns.
        # (Viz-only: ADE/FDE below already drop z.)
        pred_xyz_at_rel_ts[:, 2] = gt_future[:, 2]

        err = np.linalg.norm(pred_xyz_at_rel_ts[:, :2] - gt_future[:, :2], axis=1)
        # Compute lateral offset from target lane centerline (for viz title)
        ego_to_lane = target_lane - np.array([0.0, 0.0])  # ego at origin
        lateral_off = float(np.min(np.linalg.norm(ego_to_lane, axis=1)))
        per_frame.append({
            "frame_idx": i, "t_ns": t_i, "n_future_used": int(n),
            "rel_ts": rel_ts.tolist(),
            "lane_vid": str(target_vid), "lateral_off_m": lateral_off, "s_along_lane": 0.0,
            "pred": pred_xyz_at_rel_ts.tolist(),
            "gt_future": gt_future.tolist(),
            "ADE": float(err.mean()), "FDE": float(err[-1]),
            "skipped": False,
        })

    valid = [p for p in per_frame if not p.get("skipped")]
    ades = [p["ADE"] for p in valid]
    fdes = [p["FDE"] for p in valid]
    lane_vids = [p["lane_vid"] for p in valid]
    lane_t0 = valid[0]["lane_vid"] if valid else None
    n_same = sum(1 for v in lane_vids if v == lane_t0)
    return {
        "per_frame": per_frame,
        "n_valid": len(valid),
        "n_skipped": len(per_frame) - len(valid),
        "mean_ADE": float(np.mean(ades)) if ades else None,
        "mean_FDE": float(np.mean(fdes)) if fdes else None,
        "median_ADE": float(np.median(ades)) if ades else None,
        "median_FDE": float(np.median(fdes)) if fdes else None,
        "lane_pick_consistency": {
            "frac_same_lane": n_same / max(len(lane_vids), 1),
            "n_distinct_lanes": len(set(lane_vids)),
        },
    }


def render_one_scene(scene_full_uid: str, scene_short: str, ckpt_path: Path,
                     out_root: Path, device: str, tag: str = "planner",
                     velocity: float = 4.0, horizon_s: float = 2.0,
                     n_future: int = 4, n_panels: int = 5,
                     signvlm_jsonl: str | None = None,
                     rule_from_signvlm: bool = False):
    scene_dir = MAPDR_ROOT / scene_full_uid
    s = load_scene(scene_dir)
    print(f'\n=== {scene_short} ({scene_full_uid[:12]}) ===')

    # Load planner
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = PlannerMLP(hidden=256, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # SignVLM lanes if provided
    signvlm_lanes = None
    if signvlm_jsonl and Path(signvlm_jsonl).exists():
        signvlm_lanes = {}
        for line in open(signvlm_jsonl):
            r = json.loads(line)
            if r.get("scene_id") != scene_full_uid:
                continue
            signvlm_lanes[r["frame_idx"]] = r.get("target_lane_vec_id")
        n_valid = sum(1 for v in signvlm_lanes.values() if v is not None)
        print(f'  loaded SignVLM lanes: {len(signvlm_lanes)} frames, {n_valid} with valid lane')

    signvlm_attr = None
    if rule_from_signvlm and signvlm_jsonl:
        signvlm_attr = parse_attr_from_signvlm_jsonl(signvlm_jsonl, scene_full_uid)
        if signvlm_attr is None:
            print(f'  [rule-from-signvlm] no parseable attr_info, falling back to GT label')

    result = planner_short_window(s, model, device,
                                  horizon_s=horizon_s, n_future=n_future,
                                  signvlm_lanes=signvlm_lanes,
                                  signvlm_attr=signvlm_attr)
    print(f'  windows={len(result["per_frame"])}  valid={result["n_valid"]}')
    print(f'  mean ADE={result["mean_ADE"]:.3f}m  mean FDE={result["mean_FDE"]:.3f}m')
    lc = result["lane_pick_consistency"]
    print(f'  lane consistency: {lc["frac_same_lane"]:.0%}')

    scene_out = out_root / scene_short
    method_dir = scene_out / tag
    method_dir.mkdir(parents=True, exist_ok=True)

    json.dump({
        "scene": scene_full_uid, "method": tag,
        "velocity_ms": velocity, "horizon_s": horizon_s, "n_future": n_future,
        "mean_ADE": result["mean_ADE"], "mean_FDE": result["mean_FDE"],
        "median_ADE": result["median_ADE"], "median_FDE": result["median_FDE"],
        "lane_pick_consistency": result["lane_pick_consistency"],
        "n_windows": len(result["per_frame"]),
        "n_valid": result["n_valid"], "n_skipped": result["n_skipped"],
        "per_frame": [{k: v for k, v in p.items() if k not in ("pred", "gt_future")}
                      for p in result["per_frame"]],
    }, open(method_dir / "metrics.json", "w"), indent=2)

    plot_bev_closedloop(s, result, scene_full_uid, velocity, horizon_s,
                        method_dir / "overview_bev.png")
    plot_bev_mosaic(s, result, scene_full_uid, velocity, horizon_s,
                    method_dir / "overview_mosaic.png", n_panels=n_panels)
    plot_perframe_metrics(result, scene_full_uid,
                          method_dir / "overview_perframe.png")
    plot_bev_per_frame(s, result, scene_full_uid, velocity, horizon_s,
                       method_dir / "bev")
    plot_dashcam_per_frame(s, scene_dir, result, method_dir / "dashcam")
    print(f'  → {method_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default=None, help='full uid (or use --all-demo)')
    ap.add_argument('--all-demo', action='store_true',
                    help='render all 8 demo scenes from experiments/trajectory_smoke/')
    ap.add_argument('--ckpt', default='runs/planner/v1_100ep_n20/best.pt')
    ap.add_argument('--tag', default='planner')
    ap.add_argument('--out-root', default='experiments/trajectory_smoke')
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--velocity', type=float, default=4.0)
    ap.add_argument('--horizon-s', type=float, default=2.0)
    ap.add_argument('--n-future', type=int, default=4)
    ap.add_argument('--n-panels', type=int, default=5)
    ap.add_argument('--signvlm-jsonl', default=None)
    ap.add_argument('--rule-from-signvlm', action='store_true',
                    help='Take rule vector from SignVLM output (pred_text in jsonl) instead of GT label')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ckpt = Path(args.ckpt)

    if args.all_demo:
        # Pass DEMO_SCENE_UID_MAP=/path/to/demo_scene_uid_map.json to override.
        demo_map_path = Path(os.environ.get('DEMO_SCENE_UID_MAP', '/tmp/demo_scene_uid_map.json'))
        demo_map = json.loads(demo_map_path.read_text())
        for prefix, uid in demo_map.items():
            render_one_scene(uid, prefix, ckpt, out_root, args.device,
                             tag=args.tag, velocity=args.velocity,
                             horizon_s=args.horizon_s, n_future=args.n_future,
                             n_panels=args.n_panels, signvlm_jsonl=args.signvlm_jsonl,
                             rule_from_signvlm=args.rule_from_signvlm)
    else:
        if not args.scene:
            raise SystemExit('must pass --scene <uid> or --all-demo')
        scene_short = args.scene[:8]
        render_one_scene(args.scene, scene_short, ckpt, out_root, args.device,
                         tag=args.tag, velocity=args.velocity,
                         horizon_s=args.horizon_s, n_future=args.n_future,
                         n_panels=args.n_panels, signvlm_jsonl=args.signvlm_jsonl,
                         rule_from_signvlm=args.rule_from_signvlm)


if __name__ == '__main__':
    main()
