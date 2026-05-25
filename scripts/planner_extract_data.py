"""Day 1: Extract planner training dataset from MapDR Train split.

Per scene, save a (feature_dict, gt_trajectory) pair:
- feature_dict: structured inputs that planner consumes (see Sec "Schema" below)
- gt_trajectory: 30-frame future ego (x, y) in ego frame [m]

Schema (planner input feature_dict):
- ego_history:    (8, 4) — last 8 frames of ego state in ego frame at t=0: [x, y, yaw, v]
                  (first 8 frames of the trajectory; the planner only sees these as "history",
                   predicts the next 30 frames)
- target_lane:    (20, 2) — 20 evenly-sampled (x, y) along the rule-bound lane centerline in ego frame
- scene_lanes:    (6, 20, 2) — up to 6 other centerlines in scene (padded with zeros if fewer)
                  + scene_lanes_mask: (6,) boolean, 1 if lane present
- rule:           (10,) — flat rule constraints:
                    [0]: speed_limit (numeric, normalized by 120 km/h; -1 = N/A)
                    [1:5]: direction one-hot [TurnLeft, TurnRight, GoStraight, UTurn] (none ⇒ all 0)
                    [5]: time-active flag (1 if rule active in current frame's time window)
                    [6:10]: vehicle_allowed one-hot [Bus, Truck, NonMotor, Any]

Output gt_trajectory: (30, 2) — future ego (x, y) in ego frame at t=0 [m, m]

Saved as: data/planner_data/{split}/{uid}.pt (torch tensor dict)

Override the I/O locations via env vars:
    MAPDR_ROOT, SPLIT_JSON, PLANNER_DATA_ROOT
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
MAPDR_ROOT = Path(os.environ.get('MAPDR_ROOT', str(REPO / 'data' / 'MapDR'))).resolve()
# split.json sits alongside the resolved MapDR root by default (matches src/dataset.py)
SPLIT_JSON = Path(os.environ.get('SPLIT_JSON', str(MAPDR_ROOT.parent / 'split.json')))
OUT_ROOT = Path(os.environ.get('PLANNER_DATA_ROOT', str(REPO / 'data' / 'planner_data')))

# Constants matching schema
N_HISTORY = 8
N_FUTURE = 20  # ~5s @ 4Hz; lowered from 30 to recover scenes with shorter trajectories (32-37 frames)
N_TARGET_LANE_PTS = 20
N_SCENE_LANES = 6
SPEED_NORM = 120.0  # km/h max for normalization
DIR_VOCAB = ['TurnLeft', 'TurnRight', 'GoStraight', 'UTurn']
VEH_VOCAB = ['Bus', 'Truck', 'NonMotor', 'Any']


def pose_world_to_ego(world_xyz: np.ndarray, ego_tvec: np.ndarray, ego_R_world: np.ndarray) -> np.ndarray:
    """Convert (N, 3) world ENU points to ego frame (x forward, y left)."""
    delta = world_xyz - ego_tvec  # (N, 3)
    return delta @ ego_R_world  # (N, 3)


def quat_to_yaw(quat_xyzw: list) -> float:
    """Extract yaw from quaternion (xyzw convention used by MapDR)."""
    return R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)[2]


def extract_scene(uid: str) -> Optional[dict]:
    """Return planner feature_dict + gt_trajectory for one scene, or None if reject."""
    data_p = MAPDR_ROOT / uid / 'data.json'
    label_p = MAPDR_ROOT / uid / 'label.json'
    if not data_p.exists() or not label_p.exists():
        return None
    data = json.loads(data_p.read_text())
    label = json.loads(label_p.read_text())

    # === ego trajectory ===
    pose_dict = data['camera_pose']
    ts_sorted = sorted(pose_dict.keys(), key=int)
    if len(ts_sorted) < N_HISTORY + N_FUTURE:
        return None  # too short
    # use first 8 as history, frames 8..37 as GT future (skip rest)
    history_ts = ts_sorted[:N_HISTORY]
    future_ts = ts_sorted[N_HISTORY:N_HISTORY + N_FUTURE]

    # Anchor: ego frame defined at t=0 (last history frame)
    t0 = history_ts[-1]
    t0_pose = pose_dict[t0]
    t0_tvec = np.array(t0_pose['tvec_enu'], dtype=np.float64)
    t0_rvec = t0_pose['rvec_enu']  # quat xyzw
    # Rotation matrix world→ego: pose stores camera-to-world, we want world-to-ego
    R_cam2world = R.from_quat(t0_rvec).as_matrix()  # (3, 3)
    R_world2ego = R_cam2world.T  # world→ego rotation
    # In MapDR camera frame: x=right, y=down, z=forward. For ego planning we want
    # x=forward, y=left. Apply a fixed rotation: ego = M_egoFromCam @ cam
    # cam_x (right) → ego_y_neg, cam_z (forward) → ego_x, cam_y (down) → ego_z_neg
    M_egoFromCam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    # Net: world→ego = M_egoFromCam @ world→cam = M_egoFromCam @ R_world2ego
    R_world2ego = M_egoFromCam @ R_world2ego

    def pose_xyz_in_ego_frame(ts: str) -> np.ndarray:
        p = pose_dict[ts]
        return R_world2ego @ (np.array(p['tvec_enu']) - t0_tvec)

    # ego_history: 8 × 4 (x, y, yaw, v)
    history_xyz = np.stack([pose_xyz_in_ego_frame(ts) for ts in history_ts])  # (8, 3)
    history_xy = history_xyz[:, :2]  # (8, 2) — drop z
    # yaw relative to t=0
    history_yaw = np.zeros(N_HISTORY)
    for i, ts in enumerate(history_ts):
        rvec_i = R.from_quat(pose_dict[ts]['rvec_enu']).as_matrix()
        R_world2ego_i = M_egoFromCam @ rvec_i.T
        # yaw of frame i in ego_at_t0 frame
        R_i_in_ego = R_world2ego @ rvec_i  # rotation of frame i expressed in ego frame
        history_yaw[i] = np.arctan2(R_i_in_ego[1, 0], R_i_in_ego[0, 0])
    # velocity (finite diff, m/s)
    history_v = np.zeros(N_HISTORY)
    ts_ns = np.array([int(ts) for ts in history_ts])
    dt_s = np.diff(ts_ns).astype(np.float64) / 1e9
    if (dt_s > 0).all():
        v = np.linalg.norm(np.diff(history_xy, axis=0), axis=1) / dt_s
        history_v[1:] = v
        history_v[0] = v[0]  # fill first with second's velocity
    ego_history = np.concatenate([history_xy, history_yaw[:, None], history_v[:, None]], axis=1)  # (8, 4)

    # gt_trajectory: 30 × 2 (x, y) in ego frame
    gt_xy = np.stack([pose_xyz_in_ego_frame(ts)[:2] for ts in future_ts])  # (30, 2)

    # === scene centerlines (type='3') ===
    all_centerlines = []  # list of (N, 2) arrays
    vid_to_idx = {}
    for vid_str, vec in data['vector'].items():
        if str(vec.get('type')) != '3':
            continue
        pts_world = np.array(vec['vec_geo'], dtype=np.float64)  # (N, 3)
        pts_ego = (pts_world - t0_tvec) @ R_world2ego.T  # (N, 3)
        if pts_ego.shape[0] < 2:
            continue
        # Resample to N_TARGET_LANE_PTS along arclength
        seg = np.diff(pts_ego[:, :2], axis=0)
        cumlen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        if cumlen[-1] < 1e-6:
            continue
        t_target = np.linspace(0.0, cumlen[-1], N_TARGET_LANE_PTS)
        resampled = np.stack([
            np.interp(t_target, cumlen, pts_ego[:, 0]),
            np.interp(t_target, cumlen, pts_ego[:, 1]),
        ], axis=1)  # (20, 2)
        all_centerlines.append(resampled)
        vid_to_idx[int(vid_str)] = len(all_centerlines) - 1

    if not all_centerlines:
        return None

    # === rule (use first rule in label) ===
    rule_keys = list(label.keys())
    if not rule_keys:
        return None
    rule_first = label[rule_keys[0]]
    attr = rule_first.get('attr_info', {})
    centerline_vids = rule_first.get('centerline', [])
    if not centerline_vids:
        return None
    target_vid = centerline_vids[0]
    if target_vid not in vid_to_idx:
        return None

    target_lane = all_centerlines[vid_to_idx[target_vid]]  # (20, 2)

    # scene_lanes: pick up to N_SCENE_LANES centerlines OTHER than target (closest by midpoint dist to ego)
    other_lanes = [cl for i, cl in enumerate(all_centerlines) if i != vid_to_idx[target_vid]]
    # sort by midpoint distance to ego origin
    other_lanes.sort(key=lambda cl: np.linalg.norm(cl[N_TARGET_LANE_PTS // 2]))
    scene_lanes_padded = np.zeros((N_SCENE_LANES, N_TARGET_LANE_PTS, 2), dtype=np.float32)
    scene_mask = np.zeros(N_SCENE_LANES, dtype=np.float32)
    for i, cl in enumerate(other_lanes[:N_SCENE_LANES]):
        scene_lanes_padded[i] = cl
        scene_mask[i] = 1.0

    # === rule constraints (10-D) ===
    rule_vec = np.zeros(10, dtype=np.float32)
    # speed_limit: prefer HighSpeedLimit, fall back to LowSpeedLimit, else -1
    sl_str = str(attr.get('HighSpeedLimit', 'None'))
    if sl_str in ('None', 'N/A', 'null'):
        sl_str = str(attr.get('LowSpeedLimit', 'None'))
    try:
        rule_vec[0] = float(sl_str) / SPEED_NORM
    except (ValueError, TypeError):
        rule_vec[0] = -1.0
    # direction one-hot
    dir_field = attr.get('LaneDirection', [])
    if isinstance(dir_field, str):
        dir_field = [dir_field]
    for d in dir_field:
        if d in DIR_VOCAB:
            rule_vec[1 + DIR_VOCAB.index(d)] = 1.0
    # time-active: simplification — set 1 if EffectiveDate is not "None"
    eff_date = str(attr.get('EffectiveDate', 'None'))
    rule_vec[5] = 0.0 if eff_date in ('None', 'N/A', '') else 1.0
    # vehicle allowed one-hot
    veh_field = attr.get('AllowedTransport', 'None')
    if isinstance(veh_field, str):
        veh_field = [veh_field]
    for v in veh_field:
        if v in VEH_VOCAB:
            rule_vec[6 + VEH_VOCAB.index(v)] = 1.0
    # if no vehicle restriction, mark Any
    if rule_vec[6:10].sum() == 0:
        rule_vec[6 + VEH_VOCAB.index('Any')] = 1.0

    return {
        'uid': uid,
        'ego_history': torch.from_numpy(ego_history.astype(np.float32)),       # (8, 4)
        'target_lane': torch.from_numpy(target_lane.astype(np.float32)),       # (20, 2)
        'scene_lanes': torch.from_numpy(scene_lanes_padded),                   # (6, 20, 2)
        'scene_mask':  torch.from_numpy(scene_mask),                           # (6,)
        'rule':        torch.from_numpy(rule_vec),                             # (10,)
        'gt_trajectory': torch.from_numpy(gt_xy.astype(np.float32)),           # (30, 2)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('Train', 'Test'), default='Train')
    ap.add_argument('--limit', type=int, default=0, help='Spike: process only first N scenes')
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    args = ap.parse_args()

    split_uids = json.loads(SPLIT_JSON.read_text())[args.split]
    if args.limit:
        split_uids = split_uids[:args.limit]

    out_dir = Path(args.out_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected = []

    for uid in tqdm(split_uids, desc=f'extract {args.split}'):
        out_p = out_dir / f'{uid}.pt'
        if out_p.exists():
            continue
        sample = extract_scene(uid)
        if sample is None:
            rejected.append(uid)
            continue
        torch.save(sample, out_p)

    print(f'\n[extract] split={args.split}  saved={len(split_uids) - len(rejected)}  rejected={len(rejected)}')
    if rejected:
        with open(out_dir.parent / f'_rejected_{args.split}.json', 'w') as f:
            json.dump(rejected, f, indent=2)


if __name__ == '__main__':
    main()
