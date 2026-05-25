"""3D ENU → 2D image projection for MapDR scenes.

Each MapDR scene's `data.json` provides:
  - camera_intrinsic_matrix:  3x3 K (OpenCV convention: fx, fy, cx, cy)
  - camera_pose[timestamp]:   {"tvec_enu": [x,y,z], "rvec_enu": [qx,qy,qz,qw or qw,qx,qy,qz]}
  - traffic_board_pose:       4 ENU points (XYZ in meters)
  - vector[i].vec_geo:        N ENU points along a lane geometry

This module projects ENU points → image pixels for visualization, GT 2D label
generation, and best-frame selection (largest in-view board area).
"""
from __future__ import annotations
from typing import Iterable
import numpy as np


def quat_to_rot(q: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Unit quaternion → 3x3 rotation matrix.

    order="xyzw" (scipy/ROS default) or "wxyz" (Hamilton/Eigen).
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(f"unknown quat order: {order}")
    # Standard rotation from unit quaternion (right-handed).
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def project_points(
    points_enu: np.ndarray,
    K: np.ndarray,
    tvec_enu: np.ndarray,
    rvec_enu: np.ndarray,
    quat_order: str = "xyzw",
    pose_convention: str = "cam_to_world",
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) ENU points to (N, 2) image pixels.

    Parameters
    ----------
    pose_convention : "cam_to_world" if (R, t) maps camera frame → world,
                      "world_to_cam" if (R, t) maps world → camera.
                      MapDR appears to be cam_to_world (verified via test_projection.py).

    Returns
    -------
    uv : (N, 2) pixel coords
    depth : (N,) z in camera frame (>0 means in front of camera)
    """
    points_enu = np.asarray(points_enu, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    t = np.asarray(tvec_enu, dtype=np.float64).reshape(3)
    R = quat_to_rot(rvec_enu, quat_order)

    if pose_convention == "cam_to_world":
        # P_cam = R^T (P_world - t)
        P_cam = (points_enu - t) @ R       # equivalent to (R.T @ (P-t).T).T
    elif pose_convention == "world_to_cam":
        P_cam = points_enu @ R.T + t
    else:
        raise ValueError(f"unknown pose_convention: {pose_convention}")

    depth = P_cam[:, 2]
    # Avoid div-by-zero; keep depth sign for filtering downstream.
    safe_z = np.where(np.abs(depth) < 1e-9, np.sign(depth + 1e-12) * 1e-9, depth)
    uv = (P_cam @ K.T)[:, :2] / safe_z[:, None]
    return uv, depth


def project_polygon(
    points_3d: np.ndarray,
    K: np.ndarray,
    pose: dict,
    quat_order: str = "xyzw",
    pose_convention: str = "cam_to_world",
    img_size: tuple[int, int] | None = None,
) -> dict:
    """Project a 3D polygon (e.g. traffic_board_pose 4 points) to one frame.

    Returns
    -------
    {
      "uv":        (N, 2) projected pixels,
      "depth":     (N,) z in camera frame,
      "in_front":  bool — all points in front of camera,
      "bbox":      (xmin, ymin, xmax, ymax),
      "area_px":   bbox area in pixels (0 if not in front),
      "center_in_image": bool — bbox center inside img_size if provided,
    }
    """
    uv, depth = project_points(points_3d, K, pose["tvec_enu"], pose["rvec_enu"],
                               quat_order, pose_convention)
    in_front = bool(np.all(depth > 0))
    bbox = (float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max()))
    area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if in_front else 0.0

    center_in_image = False
    if img_size is not None and in_front:
        W, H = img_size
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        center_in_image = bool(0 <= cx <= W and 0 <= cy <= H)

    return {"uv": uv, "depth": depth, "in_front": in_front,
            "bbox": bbox, "area_px": float(area),
            "center_in_image": center_in_image}


def best_board_frame(
    board_3d: np.ndarray,
    K: np.ndarray,
    poses: dict,
    img_size: tuple[int, int],
    quat_order: str = "xyzw",
    pose_convention: str = "cam_to_world",
) -> tuple[str | None, dict | None]:
    """Find frame timestamp where the board projection has largest in-image bbox area."""
    best_ts = None
    best = None
    for ts, pose in poses.items():
        proj = project_polygon(board_3d, K, pose, quat_order, pose_convention, img_size)
        if not proj["in_front"] or not proj["center_in_image"]:
            continue
        if best is None or proj["area_px"] > best["area_px"]:
            best_ts = ts
            best = proj
    return best_ts, best


def project_polyline(
    points_enu: np.ndarray,
    K: np.ndarray,
    pose: dict,
    img_size: tuple[int, int],
    quat_order: str = "xyzw",
    pose_convention: str = "cam_to_world",
    max_depth: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a polyline's ENU points to image and keep only points that are
    in front of the camera, within max_depth, and inside the image.

    Returns
    -------
    uv_valid : (M, 2) int32 pixel coords (M <= N)
    valid_mask : (N,) bool — which input points survived filtering
    """
    W, H = img_size
    uv, depth = project_points(points_enu, K, pose["tvec_enu"], pose["rvec_enu"],
                               quat_order, pose_convention)
    in_front = depth > 0
    near_enough = depth < max_depth
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    valid = in_front & near_enough & in_image
    return uv[valid].astype(np.int32), valid


def centerline_relative_indices(vectors: dict) -> tuple[list[str], dict[str, int]]:
    """Extract type='3' centerlines from data.json `vector` dict and assign
    a 0-based relative index to each, sorted by absolute vector id.

    Returns
    -------
    ordered_abs_ids : list of absolute vector ids (str), sorted ascending as int
    abs_to_rel : dict mapping absolute id (str) → relative index (int)
    """
    centerline_ids = [vid for vid, v in vectors.items() if str(v.get("type")) == "3"]
    centerline_ids.sort(key=lambda x: int(x))
    abs_to_rel = {vid: i for i, vid in enumerate(centerline_ids)}
    return centerline_ids, abs_to_rel


def _draw_metadata_panel(
    img_bgr: np.ndarray,
    panel: dict,
    *,
    height_frac: float = 0.10,
    bg_alpha: float = 0.55,
    title: str = "[ MAP INFO ]",
) -> None:
    """Draw a HUD-style metadata panel at the bottom of `img_bgr` in-place.

    Placed at bottom rather than top to avoid overlap with overhead gantry
    signs that occupy the top of dash-cam images.

    `panel` is a dict of label → value strings. Order is preserved.
    Missing / None values are rendered as "N/A".
    """
    import cv2

    H, W = img_bgr.shape[:2]
    panel_h = max(80, int(H * height_frac))
    y0 = H - panel_h

    bg = img_bgr[y0:, :].astype(np.float32)
    img_bgr[y0:, :] = (bg * (1.0 - bg_alpha)).astype(img_bgr.dtype)

    cv2.putText(img_bgr, title, (12, y0 + int(panel_h * 0.35)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)

    items = [(k, "N/A" if v is None else str(v)) for k, v in panel.items()]
    text = "  |  ".join(f"{k}={v}" for k, v in items)
    cv2.putText(img_bgr, text, (12, y0 + int(panel_h * 0.80)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def render_visual_prompt(
    img_bgr: np.ndarray,
    vectors: dict,
    K: np.ndarray,
    pose: dict,
    quat_order: str = "xyzw",
    pose_convention: str = "cam_to_world",
    max_depth: float = 50.0,
    line_color: tuple[int, int, int] = (0, 0, 255),  # BGR red
    line_thickness: int = 10,
    font_scale: float = 1.5,
    text_color: tuple[int, int, int] = (0, 255, 255),  # BGR yellow
    text_thickness: int = 3,
    abs_to_rel: dict[str, int] | None = None,
    metadata_panel: dict | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Draw all type='3' centerlines on `img_bgr` with their relative-index
    label (0, 1, 2, ...) at the polyline's first valid projected point.

    Parameters mirror RuleVLM gen_data.gen_visual_prompt defaults
    (line thickness 10, red BGR, yellow text, scale ~1.5).

    Pass `abs_to_rel` to use a caller-supplied (e.g. shuffled) index mapping;
    if None, indices are assigned by sorted absolute vector id.

    If `metadata_panel` is provided (dict of label→value), a HUD-style overlay
    panel is rendered at the top of the image, displaying what the HD-Map says.
    Used in conflict-aware training to make the map's claim visible to the VLM
    so it can be compared against the cropped_sign content. When None, output
    is identical to the pre-CAVP behavior.

    Returns
    -------
    out_bgr : copy of img_bgr with annotations drawn
    abs_to_rel : the absolute → relative index map used for labels
    """
    import cv2

    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    if abs_to_rel is None:
        _, abs_to_rel = centerline_relative_indices(vectors)

    for abs_id, rel_idx in abs_to_rel.items():
        pts_3d = np.asarray(vectors[abs_id]["vec_geo"], dtype=np.float64)
        uv_valid, _ = project_polyline(pts_3d, K, pose, (W, H),
                                       quat_order, pose_convention, max_depth)
        if len(uv_valid) < 2:
            continue
        cv2.polylines(out, [uv_valid.reshape(-1, 1, 2)], isClosed=False,
                      color=line_color, thickness=line_thickness)
        u0, v0 = int(uv_valid[0, 0]), int(uv_valid[0, 1])
        cv2.putText(out, str(rel_idx), (u0, v0), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, text_thickness, cv2.LINE_AA)

    if metadata_panel is not None:
        _draw_metadata_panel(out, metadata_panel)

    return out, abs_to_rel


def panel_from_attr_info(rules: list, max_rules: int = 4) -> dict:
    """Build a metadata panel dict from a list of MapDR `attr_info` records.

    Aggregates LaneType / speed / direction / vehicle / time across up to
    `max_rules` rules. Returns a 5-key dict suitable for `metadata_panel`.

    Schema matches `src/dataset.py:ATTR_KEYS`:
      AllowedTransport, EffectiveDate, EffectiveTime, HighSpeedLimit,
      LaneDirection, LaneType, LowSpeedLimit, RuleIndex

    Designed for the *clean* (faithful) case where the panel mirrors the GT.
    For conflict perturbation, build a perturbed panel from this baseline
    and pass that in instead.
    """
    def _collect(field):
        vals = []
        for r in rules[:max_rules]:
            v = r.get(field)
            if v in (None, "None", "", []):
                continue
            if isinstance(v, list):
                vals.extend(str(x) for x in v if x not in (None, "None", ""))
            else:
                vals.append(str(v))
        return vals

    def _summarize(vals, sep="/"):
        if not vals:
            return "N/A"
        seen, ordered = set(), []
        for v in vals:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        return sep.join(ordered)

    lane_type = _summarize(_collect("LaneType"))

    speed = _summarize(_collect("HighSpeedLimit"))
    if speed != "N/A":
        low = _summarize(_collect("LowSpeedLimit"))
        if low != "N/A":
            speed = f"{low}-{speed}"

    direction = _summarize(_collect("LaneDirection"), sep=",")

    allowed = _collect("AllowedTransport")
    if not allowed or set(allowed) == {"Vehicle"}:
        vehicle = "any"
    else:
        vehicle = _summarize(allowed, sep=",")

    time_range = _summarize(_collect("EffectiveTime"), sep=",")
    if time_range == "N/A":
        time_range = "24h"

    return {
        "Type": lane_type,
        "Limit": speed,
        "Dir": direction,
        "Veh": vehicle,
        "Time": time_range,
    }
