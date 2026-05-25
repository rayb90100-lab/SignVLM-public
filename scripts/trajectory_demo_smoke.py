"""Trajectory demo smoke v2 — short-window closed-loop baseline.

Framework: at each frame t_i with future window of `horizon_s` seconds:
  1. take current ego pose
  2. pick nearest lane by lateral offset (re-pick every frame)
  3. project ego to lane centerline → s_i
  4. unroll: s_i + v * τ for τ ∈ linspace(0, horizon_s, n_future)
  5. compare predicted (n_future, 3) to GT [t_{i+1}, ..., t_{i+n_future}]
  6. accumulate ADE/FDE per window, mean across all frames

Matches nuScenes / Argoverse motion forecasting evaluation protocol.

Usage:
  python scripts/trajectory_demo_smoke.py
  python scripts/trajectory_demo_smoke.py --scene <hash> --horizon-s 2.0 --velocity 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# allow `from src.projection import ...` when run as `python scripts/foo.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.projection import project_points, quat_to_rot

MAPDR_ROOT_DEFAULT = str(Path(__file__).resolve().parent.parent / "data" / "MapDR")


# ---------- data loading ----------

def load_scene(scene_dir: Path) -> dict:
    data = json.load(open(scene_dir / "data.json"))
    label = json.load(open(scene_dir / "label.json"))
    return {
        "poses": data["camera_pose"],
        "vectors": data["vector"],
        "K": np.array(data["camera_intrinsic_matrix"]),
        "label": label,
        "board": np.array(data["traffic_board_pose"]),
    }


def gt_trajectory(poses: dict) -> tuple[list[int], np.ndarray]:
    ts = sorted(poses.keys(), key=int)
    enu = np.array([poses[t]["tvec_enu"] for t in ts], dtype=np.float64)
    return [int(t) for t in ts], enu


# ---------- lane geometry ----------

def all_centerlines(vectors: dict) -> list[tuple[str, np.ndarray]]:
    """All type=3 vectors as polylines."""
    out = []
    for vid, v in vectors.items():
        if str(v.get("type")) == "3":
            out.append((vid, np.asarray(v["vec_geo"], dtype=np.float64)))
    return out


def arc_lengths(polyline: np.ndarray) -> np.ndarray:
    if len(polyline) < 2:
        return np.array([0.0])
    diff = np.diff(polyline, axis=0)
    seg = np.linalg.norm(diff, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_point_to_polyline(p: np.ndarray, polyline: np.ndarray,
                              arc: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Return (arc_s, foot_xyz, lateral_dist) — closest point on polyline."""
    p = np.asarray(p, dtype=np.float64)
    best_s, best_foot, best_d = 0.0, polyline[0].copy(), np.inf
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            continue
        t = np.dot(p - a, ab) / L2
        t = float(np.clip(t, 0.0, 1.0))
        foot = a + t * ab
        d = float(np.linalg.norm(p - foot))
        if d < best_d:
            best_d = d
            best_foot = foot
            best_s = float(arc[i] + t * np.linalg.norm(ab))
    return best_s, best_foot, best_d


def arc_to_xyz(s_target: float, polyline: np.ndarray,
               arc: np.ndarray) -> np.ndarray:
    s_total = arc[-1]
    if s_target <= 0:
        return polyline[0].copy()
    if s_target >= s_total:
        last_dir = polyline[-1] - polyline[-2]
        last_len = float(np.linalg.norm(last_dir))
        if last_len < 1e-9:
            return polyline[-1].copy()
        return polyline[-1] + (s_target - s_total) * last_dir / last_len
    i = int(np.searchsorted(arc, s_target, side="right") - 1)
    i = max(0, min(i, len(polyline) - 2))
    seg_len = arc[i + 1] - arc[i]
    if seg_len < 1e-9:
        return polyline[i].copy()
    t = (s_target - arc[i]) / seg_len
    return polyline[i] + t * (polyline[i + 1] - polyline[i])


def pick_nearest_lane(ego_xyz: np.ndarray,
                      centerlines: list[tuple[str, np.ndarray]]) -> tuple[str, np.ndarray, float]:
    """Find centerline minimizing lateral offset to ego_xyz."""
    best_vid, best_poly, best_d = None, None, np.inf
    for vid, poly in centerlines:
        if len(poly) < 2:
            continue
        arc = arc_lengths(poly)
        _, _, d = project_point_to_polyline(ego_xyz, poly, arc)
        if d < best_d:
            best_d, best_vid, best_poly = d, vid, poly
    return best_vid, best_poly, best_d


# ---------- short-window closed-loop baseline ----------

def baseline_short_window(scene: dict, horizon_s: float = 2.0,
                          n_future: int = 4, velocity: float = 4.0,
                          signvlm_lanes: dict[int, str] | None = None) -> dict:
    """Per-frame closed-loop. Returns per-frame metrics + means.

    If signvlm_lanes is None: geometric baseline (pick_nearest_lane).
    Else: use signvlm_lanes[frame_idx] = vec_id (skip frame if lane missing).
    """
    ts, gt = gt_trajectory(scene["poses"])
    centerlines = all_centerlines(scene["vectors"])
    if not centerlines:
        return {"per_frame": [], "mean_ADE": None, "mean_FDE": None}
    vectors_by_vid = {vid: poly for vid, poly in centerlines}

    per_frame = []
    for i in range(len(ts)):
        t_i = ts[i]
        ego_xyz = gt[i]
        future_horizon_ns = horizon_s * 1e9
        future_ts = [t for t in ts[i+1:] if (t - t_i) <= future_horizon_ns]
        if len(future_ts) < 1:
            continue
        n = min(n_future, len(future_ts))
        idxs = np.linspace(0, len(future_ts) - 1, num=n).astype(int)
        sampled_ts = [future_ts[j] for j in idxs]
        gt_future = np.array([scene["poses"][str(t)]["tvec_enu"]
                              for t in sampled_ts], dtype=np.float64)
        rel_ts = np.array([(t - t_i) / 1e9 for t in sampled_ts])

        # Lane choice
        if signvlm_lanes is not None:
            lane_vid = signvlm_lanes.get(i)
            if lane_vid is None or lane_vid not in vectors_by_vid:
                # SignVLM 没给出 valid lane → 标记 + 跳过 (mean stats 排除)
                per_frame.append({
                    "frame_idx": i, "t_ns": t_i,
                    "lane_vid": lane_vid, "lateral_off_m": None,
                    "s_along_lane": None,
                    "pred": [], "gt_future": gt_future.tolist(),
                    "ADE": None, "FDE": None, "skipped": True,
                })
                continue
            poly = vectors_by_vid[lane_vid]
        else:
            lane_vid, poly, _ = pick_nearest_lane(ego_xyz, centerlines)
        arc = arc_lengths(poly)
        s_i, foot, lat_off = project_point_to_polyline(ego_xyz, poly, arc)
        pred = np.array([arc_to_xyz(s_i + velocity * tau, poly, arc)
                         for tau in rel_ts])
        err = np.linalg.norm(pred[:, :2] - gt_future[:, :2], axis=1)
        per_frame.append({
            "frame_idx": i, "t_ns": t_i, "n_future_used": int(n),
            "rel_ts": rel_ts.tolist(),
            "lane_vid": lane_vid, "lateral_off_m": float(lat_off),
            "s_along_lane": float(s_i),
            "pred": pred.tolist(), "gt_future": gt_future.tolist(),
            "ADE": float(err.mean()), "FDE": float(err[-1]),
            "skipped": False,
        })

    valid = [p for p in per_frame if not p.get("skipped")]
    ades = [p["ADE"] for p in valid]
    fdes = [p["FDE"] for p in valid]
    return {
        "per_frame": per_frame,
        "n_valid": len(valid),
        "n_skipped": len(per_frame) - len(valid),
        "mean_ADE": float(np.mean(ades)) if ades else None,
        "mean_FDE": float(np.mean(fdes)) if fdes else None,
        "median_ADE": float(np.median(ades)) if ades else None,
        "median_FDE": float(np.median(fdes)) if fdes else None,
        "lane_pick_consistency": _lane_consistency(valid),
    }


def _lane_consistency(per_frame: list) -> dict:
    """Fraction of frames that pick the same lane as frame 0."""
    if not per_frame:
        return {"frac_same_lane": None, "n_distinct_lanes": 0}
    first = per_frame[0]["lane_vid"]
    same = sum(1 for p in per_frame if p["lane_vid"] == first)
    distinct = len({p["lane_vid"] for p in per_frame})
    return {"frac_same_lane": same / len(per_frame),
            "n_distinct_lanes": distinct}


# ---------- visualization ----------

def to_ego_frame(pts_enu: np.ndarray, ego_xyz: np.ndarray,
                 R_ego: np.ndarray) -> np.ndarray:
    pts_cam = (pts_enu - ego_xyz) @ R_ego
    return np.column_stack([pts_cam[:, 2], -pts_cam[:, 0]])


def plot_bev_closedloop(scene: dict, result: dict, scene_hash: str,
                        velocity: float, horizon_s: float, out_path: Path):
    """BEV in ego-start frame: GT full trajectory + each prediction window."""
    if not HAS_MPL:
        return
    ts, gt = gt_trajectory(scene["poses"])
    pose0 = scene["poses"][str(ts[0])]
    ego0 = np.asarray(pose0["tvec_enu"])
    R0 = quat_to_rot(np.asarray(pose0["rvec_enu"]), "xyzw")

    gt_e = to_ego_frame(gt, ego0, R0)
    fig, ax = plt.subplots(figsize=(10, 8))
    # background lanes
    for vid, v in scene["vectors"].items():
        if str(v.get("type")) != "3":
            continue
        seg_e = to_ego_frame(np.asarray(v["vec_geo"]), ego0, R0)
        ax.plot(seg_e[:, 0], seg_e[:, 1], color="lightgray", lw=1, zorder=1)
    # GT
    ax.plot(gt_e[:, 0], gt_e[:, 1], "g-", lw=2, label="GT", zorder=4)
    ax.scatter(gt_e[:, 0], gt_e[:, 1], c="green", s=20, zorder=4)
    # Each prediction window (skip frames with no valid lane)
    valid = [p for p in result["per_frame"] if not p.get("skipped")]
    n_w = len(valid)
    cmap = plt.get_cmap("Reds")
    for k, p in enumerate(valid):
        pred = np.asarray(p["pred"])
        if pred.size == 0:
            continue
        pred_e = to_ego_frame(pred, ego0, R0)
        c = cmap(0.4 + 0.5 * k / max(n_w - 1, 1))
        ax.plot(pred_e[:, 0], pred_e[:, 1], "-", color=c, lw=1.2,
                alpha=0.85, zorder=3)
        ax.scatter([gt_e[p["frame_idx"], 0]], [gt_e[p["frame_idx"], 1]],
                   c=[c], s=8, zorder=5)
    # Mark skipped frames distinctly (X marker on ego at that frame)
    for p in result["per_frame"]:
        if p.get("skipped"):
            ax.scatter([gt_e[p["frame_idx"], 0]], [gt_e[p["frame_idx"], 1]],
                       c="gray", s=40, marker="x", zorder=5)
    ax.scatter([0], [0], c="blue", s=140, marker="*", label="ego start", zorder=6)
    ax.annotate("", xy=(5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.set_aspect("equal")
    ax.set_xlabel("forward (m)")
    ax.set_ylabel("left (m)")
    ax.set_title(f"{scene_hash[:8]}  closed-loop baseline  v={velocity:.0f} m/s "
                 f"horizon={horizon_s:.1f}s  "
                 f"mean ADE={result['mean_ADE']:.2f}  FDE={result['mean_FDE']:.2f}  "
                 f"({n_w} windows)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="lightblue", lw=0.5, zorder=0)
    ax.axvline(0, color="lightblue", lw=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  [bev] saved → {out_path}")


def plot_bev_per_frame(scene: dict, result: dict, scene_hash: str,
                       velocity: float, horizon_s: float, out_dir: Path):
    """One BEV PNG per frame in the scene, saved as frame_NN.png in out_dir."""
    if not HAS_MPL or not result["per_frame"]:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ts, gt = gt_trajectory(scene["poses"])
    pose0 = scene["poses"][str(ts[0])]
    ego0 = np.asarray(pose0["tvec_enu"])
    R0 = quat_to_rot(np.asarray(pose0["rvec_enu"]), "xyzw")
    bg = [(vid, to_ego_frame(np.asarray(v["vec_geo"]), ego0, R0))
          for vid, v in scene["vectors"].items()
          if str(v.get("type")) == "3"]
    gt_e = to_ego_frame(gt, ego0, R0)
    xlim = (float(gt_e[:, 0].min()) - 5, float(gt_e[:, 0].max()) + 10)
    ylim = (float(gt_e[:, 1].min()) - 10, float(gt_e[:, 1].max()) + 10)

    for p in result["per_frame"]:
        fi = p["frame_idx"]
        skipped = p.get("skipped", False)
        fig, ax = plt.subplots(figsize=(7, 5))
        for vid, seg_e in bg:
            ax.plot(seg_e[:, 0], seg_e[:, 1], color="lightgray", lw=0.8, zorder=1)
        ax.plot(gt_e[:, 0], gt_e[:, 1], color="lightgreen", lw=1, alpha=0.4, zorder=2)
        lane_vid = p["lane_vid"]
        if lane_vid is not None and lane_vid in scene["vectors"]:
            lane_pts = np.asarray(scene["vectors"][lane_vid]["vec_geo"])
            lane_e = to_ego_frame(lane_pts, ego0, R0)
            ax.plot(lane_e[:, 0], lane_e[:, 1], "k-", lw=2.2, zorder=3,
                    label=f"picked lane (vid={lane_vid})")
        ego_now = gt_e[fi]
        ax.scatter([ego_now[0]], [ego_now[1]],
                   c=("red" if skipped else "blue"),
                   s=120, marker="*", zorder=6,
                   label="ego" if not skipped else "ego (skipped)")
        if p.get("gt_future"):
            gtf_e = to_ego_frame(np.asarray(p["gt_future"]), ego0, R0)
            ax.plot(gtf_e[:, 0], gtf_e[:, 1], "go-", lw=2, ms=6, zorder=5,
                    label="GT future")
        if p.get("pred"):
            pred_e = to_ego_frame(np.asarray(p["pred"]), ego0, R0)
            ax.plot(pred_e[:, 0], pred_e[:, 1], "ro--", lw=2, ms=6, zorder=4,
                    label="pred future")
        if skipped:
            title = (f"{scene_hash[:8]}  f{fi:02d}/{len(ts)-1}  SKIPPED  "
                     f"lane_vid={lane_vid}")
        else:
            title = (f"{scene_hash[:8]}  f{fi:02d}/{len(ts)-1}  "
                     f"ADE={p['ADE']:.2f}m  FDE={p['FDE']:.2f}m  "
                     f"lat={p['lateral_off_m']:.2f}m  lane={lane_vid}")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("forward (m)")
        ax.set_ylabel("left (m)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{fi:02d}.png", dpi=100)
        plt.close(fig)
    print(f"  [bev/frame] {len(result['per_frame'])} pngs → {out_dir}")


def plot_bev_mosaic(scene: dict, result: dict, scene_hash: str,
                    velocity: float, horizon_s: float, out_path: Path,
                    n_panels: int = 9):
    """Grid of N BEV snapshots — closed-loop state at evenly-spaced frames.

    Each subplot shows: background lanes (gray), picked lane this frame (black),
    ego pos (blue), 2s pred future (red), 2s GT future (green).
    All in fixed ego-start frame coordinates so subplots are spatially comparable.
    """
    if not HAS_MPL or not result["per_frame"]:
        return
    ts, gt = gt_trajectory(scene["poses"])
    pose0 = scene["poses"][str(ts[0])]
    ego0 = np.asarray(pose0["tvec_enu"])
    R0 = quat_to_rot(np.asarray(pose0["rvec_enu"]), "xyzw")

    # bg lanes in ego frame
    bg = []
    for vid, v in scene["vectors"].items():
        if str(v.get("type")) != "3":
            continue
        bg.append((vid, to_ego_frame(np.asarray(v["vec_geo"]), ego0, R0)))

    # GT全程 in ego frame (background reference)
    gt_e = to_ego_frame(gt, ego0, R0)

    # Pick n_panels evenly-spaced windows
    n_w = len(result["per_frame"])
    panel_idxs = np.linspace(0, n_w - 1, num=min(n_panels, n_w)).astype(int)
    panels = [result["per_frame"][i] for i in panel_idxs]

    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    # compute global xlim/ylim from GT全程 + 一个 margin
    xlim = (float(gt_e[:, 0].min()) - 5, float(gt_e[:, 0].max()) + 10)
    ylim = (float(gt_e[:, 1].min()) - 10, float(gt_e[:, 1].max()) + 10)

    for k, p in enumerate(panels):
        ax = axes[k // ncols, k % ncols]
        fi = p["frame_idx"]
        skipped = p.get("skipped", False)
        # bg lanes
        for vid, seg_e in bg:
            ax.plot(seg_e[:, 0], seg_e[:, 1], color="lightgray", lw=0.8, zorder=1)
        # GT 整条 trace (faint, 看上下文)
        ax.plot(gt_e[:, 0], gt_e[:, 1], color="lightgreen", lw=1, alpha=0.4,
                zorder=2)
        # picked lane in this frame (full polyline highlighted)
        lane_vid = p["lane_vid"]
        if lane_vid is not None and lane_vid in scene["vectors"]:
            lane_pts = np.asarray(scene["vectors"][lane_vid]["vec_geo"])
            lane_e = to_ego_frame(lane_pts, ego0, R0)
            ax.plot(lane_e[:, 0], lane_e[:, 1], "k-", lw=2.2, zorder=3)
        # ego current pos
        ego_e_now = gt_e[fi]
        ax.scatter([ego_e_now[0]], [ego_e_now[1]],
                   c=("red" if skipped else "blue"),
                   s=80, marker="*", zorder=6)
        # GT future window (green) — always available
        if p.get("gt_future"):
            gt_fut_e = to_ego_frame(np.asarray(p["gt_future"]), ego0, R0)
            ax.plot(gt_fut_e[:, 0], gt_fut_e[:, 1], "go-", lw=2, ms=5, zorder=5)
        # Pred future window (red) — may be empty for skipped
        if p.get("pred"):
            pred_e = to_ego_frame(np.asarray(p["pred"]), ego0, R0)
            ax.plot(pred_e[:, 0], pred_e[:, 1], "ro--", lw=2, ms=5, zorder=4)
        if skipped:
            title = f"f{fi:02d}  SKIPPED  lane={lane_vid}"
        else:
            title = (f"f{fi:02d}  ADE={p['ADE']:.1f} FDE={p['FDE']:.1f} "
                     f"lat={p['lateral_off_m']:.1f}m  lane={lane_vid}")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
    # Hide unused
    for k in range(len(panels), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(
        f"{scene_hash[:8]}  closed-loop mosaic  v={velocity:.0f}m/s "
        f"horizon={horizon_s:.1f}s  mean ADE={result['mean_ADE']:.2f}m  "
        f"lane consistency={result['lane_pick_consistency']['frac_same_lane']:.0%}",
        fontsize=11)
    fig.supxlabel("forward (m, ego-start frame)")
    fig.supylabel("left (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  [mosaic] saved → {out_path}")


def plot_perframe_metrics(result: dict, scene_hash: str, out_path: Path):
    """Per-frame ADE/FDE timeline."""
    if not HAS_MPL or not result["per_frame"]:
        return
    valid = [p for p in result["per_frame"] if not p.get("skipped")]
    if not valid:
        return
    idxs = [p["frame_idx"] for p in valid]
    ades = [p["ADE"] for p in valid]
    fdes = [p["FDE"] for p in valid]
    lats = [p["lateral_off_m"] for p in valid]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(idxs, ades, "o-", label="ADE", color="tab:blue")
    axes[0].plot(idxs, fdes, "s-", label="FDE", color="tab:red")
    axes[0].axhline(result["mean_ADE"], ls="--", color="tab:blue", alpha=0.4,
                    label=f"mean ADE={result['mean_ADE']:.2f}")
    axes[0].axhline(result["mean_FDE"], ls="--", color="tab:red", alpha=0.4,
                    label=f"mean FDE={result['mean_FDE']:.2f}")
    axes[0].set_ylabel("error (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(idxs, lats, "x-", color="tab:purple", label="lateral offset")
    axes[1].set_ylabel("lateral off (m)")
    axes[1].set_xlabel("frame idx")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{scene_hash[:8]}  per-frame metrics  "
                 f"lane consistency = {result['lane_pick_consistency']['frac_same_lane']:.0%} same as t0  "
                 f"({result['lane_pick_consistency']['n_distinct_lanes']} distinct lanes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  [metrics] saved → {out_path}")


def plot_dashcam_per_frame(scene: dict, scene_dir: Path, result: dict,
                           out_dir: Path):
    """One dashcam-overlay PNG per frame, saved as frame_NN.png in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in result["per_frame"]:
        plot_dashcam_window(scene, scene_dir, result, p["frame_idx"],
                            out_dir / f"frame_{p['frame_idx']:02d}.png")
    print(f"  [dashcam/frame] {len(result['per_frame'])} pngs → {out_dir}")


def plot_dashcam_window(scene: dict, scene_dir: Path, result: dict,
                        frame_idx: int, out_path: Path):
    """Overlay prediction window on dash cam at given frame_idx."""
    ts, gt = gt_trajectory(scene["poses"])
    if frame_idx >= len(ts):
        return
    t_i = ts[frame_idx]
    img = cv2.imread(str(scene_dir / "img" / f"{t_i}.jpg"))
    if img is None:
        print(f"  [overlay] no image at frame {frame_idx}")
        return
    H, W = img.shape[:2]
    pose_i = scene["poses"][str(t_i)]
    tvec = np.asarray(pose_i["tvec_enu"])
    rvec = np.asarray(pose_i["rvec_enu"])

    def proj(pts3d):
        uv, depth = project_points(pts3d, scene["K"], tvec, rvec)
        m = ((depth > 0.5) &
             (uv[:, 0] >= -50) & (uv[:, 0] < W + 50) &
             (uv[:, 1] >= -50) & (uv[:, 1] < H + 50))
        return uv, m

    # find result for this frame
    pf = next((p for p in result["per_frame"] if p["frame_idx"] == frame_idx), None)
    if pf is None:
        return
    pred = np.asarray(pf["pred"])
    gt_fut = np.asarray(pf["gt_future"])

    pred_uv, pred_m = proj(pred)
    gt_uv, gt_m = proj(gt_fut)
    # picked lane geometry
    lane_vid = pf["lane_vid"]
    if lane_vid is not None and lane_vid in scene["vectors"]:
        lane_pts = np.asarray(scene["vectors"][lane_vid]["vec_geo"])
        lane_uv, lane_m = proj(lane_pts)
        pts = lane_uv[lane_m].astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, (200, 200, 200), 4, cv2.LINE_AA)

    # GT future window
    pts = gt_uv[gt_m].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, (0, 220, 0), 4, cv2.LINE_AA)
    for u, v in pts:
        cv2.circle(img, (int(u), int(v)), 7, (0, 220, 0), -1, cv2.LINE_AA)
    # Pred future window
    pts = pred_uv[pred_m].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, (0, 0, 230), 4, cv2.LINE_AA)
    for u, v in pts:
        cv2.circle(img, (int(u), int(v)), 7, (0, 0, 230), -1, cv2.LINE_AA)

    if pf.get("skipped"):
        title = f"frame {frame_idx}  SKIPPED  lane={pf['lane_vid']}"
    else:
        title = (f"frame {frame_idx}  ADE={pf['ADE']:.2f}m  FDE={pf['FDE']:.2f}m  "
                 f"lat_off={pf['lateral_off_m']:.2f}m")
    cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (20, 20, 20), 1, cv2.LINE_AA)
    for i, (lab, col) in enumerate([("lane", (200, 200, 200)),
                                     ("GT future", (0, 220, 0)),
                                     ("pred future", (0, 0, 230))]):
        y = 90 + i * 35
        cv2.rectangle(img, (20, y - 18), (55, y + 4), col, -1)
        cv2.putText(img, lab, (65, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    print(f"  [overlay] frame {frame_idx} saved → {out_path}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapdr-root", default=MAPDR_ROOT_DEFAULT)
    ap.add_argument("--scene", default="00002ed8da8843a680469cf65afecddd")
    ap.add_argument("--velocity", type=float, default=4.0,
                    help="constant velocity in m/s")
    ap.add_argument("--horizon-s", type=float, default=2.0,
                    help="prediction horizon in seconds (2 = nuScenes short)")
    ap.add_argument("--n-future", type=int, default=4,
                    help="number of future frames to predict in window")
    ap.add_argument("--out-dir", default="experiments/trajectory_smoke")
    ap.add_argument("--n-panels", type=int, default=9,
                    help="number of frame snapshots in mosaic grid")
    ap.add_argument("--signvlm-jsonl", default="",
                    help="path to trajectory_signvlm_infer output jsonl; if set, "
                         "uses SignVLM target_lane per frame instead of geometric heuristic")
    ap.add_argument("--tag", default="",
                    help="extra label appended to output filenames (e.g. 'C05' / 'no_perturb')")
    args = ap.parse_args()

    scene_dir = Path(args.mapdr_root) / args.scene
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== scene: {scene_dir.name} ===")
    s = load_scene(scene_dir)
    ts, gt = gt_trajectory(s["poses"])
    print(f"  frames = {len(ts)}  duration = {(ts[-1]-ts[0])/1e9:.2f} s  "
          f"chord = {np.linalg.norm(gt[-1] - gt[0]):.2f} m")
    print(f"  v = {args.velocity:.1f} m/s  horizon = {args.horizon_s:.1f} s  "
          f"n_future = {args.n_future}")

    # Optionally load SignVLM per-frame lane choices
    signvlm_lanes = None
    if args.signvlm_jsonl:
        signvlm_lanes = {}
        for line in open(args.signvlm_jsonl):
            r = json.loads(line)
            if r["scene_id"] != args.scene:
                continue
            signvlm_lanes[r["frame_idx"]] = r.get("target_lane_vec_id")
        n_valid = sum(1 for v in signvlm_lanes.values() if v is not None)
        print(f"  loaded SignVLM lanes from {args.signvlm_jsonl}: "
              f"{len(signvlm_lanes)} frames, {n_valid} with valid lane")

    result = baseline_short_window(s, horizon_s=args.horizon_s,
                                   n_future=args.n_future,
                                   velocity=args.velocity,
                                   signvlm_lanes=signvlm_lanes)
    n_w = len(result["per_frame"])
    print(f"  prediction windows = {n_w}")
    if n_w == 0:
        print("  [skip] no windows produced")
        return

    print(f"  mean ADE = {result['mean_ADE']:.3f} m  "
          f"median ADE = {result['median_ADE']:.3f} m")
    print(f"  mean FDE = {result['mean_FDE']:.3f} m  "
          f"median FDE = {result['median_FDE']:.3f} m")
    lc = result["lane_pick_consistency"]
    print(f"  lane consistency: {lc['frac_same_lane']:.0%} same lane as t0  "
          f"({lc['n_distinct_lanes']} distinct lanes picked)")

    # Per-scene subdir; per-method (tag) subdir under it
    method_tag = args.tag if args.tag else "geo"
    scene_dir_out = out_dir / args.scene[:8]
    method_dir = scene_dir_out / method_tag
    scene_dir_out.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)

    json.dump({
        "scene": args.scene, "method": method_tag,
        "velocity_ms": args.velocity, "horizon_s": args.horizon_s,
        "n_future": args.n_future,
        "mean_ADE": result["mean_ADE"], "mean_FDE": result["mean_FDE"],
        "median_ADE": result["median_ADE"], "median_FDE": result["median_FDE"],
        "lane_pick_consistency": result["lane_pick_consistency"],
        "n_windows": n_w, "n_valid": result.get("n_valid"),
        "n_skipped": result.get("n_skipped"),
        "per_frame": [{k: v for k, v in p.items() if k not in ("pred", "gt_future")}
                      for p in result["per_frame"]],
    }, open(method_dir / "metrics.json", "w"), indent=2)

    # Overview plots (scene-level summary, not per-frame)
    plot_bev_closedloop(s, result, args.scene, args.velocity, args.horizon_s,
                        method_dir / "overview_bev.png")
    plot_bev_mosaic(s, result, args.scene, args.velocity, args.horizon_s,
                    method_dir / "overview_mosaic.png",
                    n_panels=args.n_panels)
    plot_perframe_metrics(result, args.scene,
                          method_dir / "overview_perframe.png")

    # Per-frame BEV (one png per frame)
    plot_bev_per_frame(s, result, args.scene, args.velocity, args.horizon_s,
                       method_dir / "bev")

    # Per-frame dashcam overlay (one png per frame)
    plot_dashcam_per_frame(s, scene_dir, result, method_dir / "dashcam")


if __name__ == "__main__":
    main()
