"""Render per-scene visual prompt images for the zeroshot_demo set.

For each scene under data/zeroshot_demo/*:
  1. Pick the representative frame via best_board_frame (largest in-image
     traffic_board projection area)
  2. Crop the sign region (RuleVLM-style: 1.4x horiz / 1.6x vert expand)
  3. On the same frame, project all type='3' centerlines and draw red lines
     plus a yellow relative-index digit at each polyline's first valid point

Outputs land in experiments/visual_prompts/<scene_name>/:
  - cropped_sign.jpg
  - visual_prompt.jpg
  - meta.json   (representative ts, abs->rel centerline map, GT label centerlines)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from projection import (  # noqa: E402
    best_board_frame,
    project_polygon,
    render_visual_prompt,
    centerline_relative_indices,
    panel_from_attr_info,
)

ZEROSHOT_DIR = REPO / "data" / "zeroshot_demo"
OUT_DIR = REPO / "experiments" / "visual_prompts"
QUAT_ORDER = "xyzw"
POSE_CONV = "cam_to_world"


def expand_bbox(bbox: tuple[float, float, float, float],
                expand_x: float, expand_y: float,
                img_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Expand a bbox by horizontal / vertical multipliers, clipped to image."""
    W, H = img_size
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = (x1 - x0) * expand_x
    h = (y1 - y0) * expand_y
    nx0 = max(0, int(cx - w / 2))
    ny0 = max(0, int(cy - h / 2))
    nx1 = min(W, int(cx + w / 2))
    ny1 = min(H, int(cy + h / 2))
    return nx0, ny0, nx1, ny1


def process_scene(scene_dir: Path, out_dir: Path) -> dict:
    data = json.loads((scene_dir / "data.json").read_text())
    label = json.loads((scene_dir / "label.json").read_text())
    K = np.array(data["camera_intrinsic_matrix"])
    poses = data["camera_pose"]
    board = np.array(data["traffic_board_pose"])
    vectors = data["vector"]

    img_files = sorted((scene_dir / "img").glob("*.jpg"))
    if not img_files:
        return {"scene": scene_dir.name, "error": "no images"}
    W, H = cv2.imread(str(img_files[0])).shape[1::-1]

    best_ts, best = best_board_frame(board, K, poses, (W, H), QUAT_ORDER, POSE_CONV)
    if best_ts is None:
        return {"scene": scene_dir.name, "error": "no valid representative frame"}

    rep_img_path = scene_dir / "img" / f"{best_ts}.jpg"
    if not rep_img_path.exists():
        return {"scene": scene_dir.name, "error": f"missing img {rep_img_path.name}"}

    img = cv2.imread(str(rep_img_path))
    if img is None:
        return {"scene": scene_dir.name, "error": f"cv2 cannot read {rep_img_path}"}

    # 1) cropped sign  (RuleVLM expand_bbox_1: x*1.4, y*1.6)
    crop_box = expand_bbox(best["bbox"], 1.4, 1.6, (W, H))
    cx0, cy0, cx1, cy1 = crop_box
    cropped = img[cy0:cy1, cx0:cx1]

    # 2) visual prompt  (centerlines + relative-index labels)
    vp, abs_to_rel = render_visual_prompt(
        img, vectors, K, poses[best_ts], QUAT_ORDER, POSE_CONV
    )
    # Also draw the board bbox for context (thin green rectangle)
    bx0, by0, bx1, by1 = (int(v) for v in best["bbox"])
    cv2.rectangle(vp, (bx0, by0), (bx1, by1), (0, 255, 0), 3)

    # 2b) visual prompt with metadata panel (CAVP: HD-Map says what)
    rules_attr = [r.get("attr_info", {}) for r in label.values()]
    panel = panel_from_attr_info(rules_attr)
    vp_panel, _ = render_visual_prompt(
        img, vectors, K, poses[best_ts], QUAT_ORDER, POSE_CONV,
        metadata_panel=panel,
    )
    cv2.rectangle(vp_panel, (bx0, by0), (bx1, by1), (0, 255, 0), 3)

    # 3) write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "cropped_sign.jpg"), cropped)
    cv2.imwrite(str(out_dir / "visual_prompt.jpg"), vp)
    cv2.imwrite(str(out_dir / "visual_prompt_with_panel.jpg"), vp_panel)

    # GT centerline mapping (absolute -> relative) for sanity
    gt_centerlines_abs = {rid: r["centerline"] for rid, r in label.items()}
    gt_centerlines_rel = {
        rid: [abs_to_rel[str(c)] for c in clist if str(c) in abs_to_rel]
        for rid, clist in gt_centerlines_abs.items()
    }

    meta = {
        "scene": scene_dir.name,
        "rep_ts": best_ts,
        "rep_img": str(rep_img_path.relative_to(REPO)) if rep_img_path.is_relative_to(REPO) else str(rep_img_path),
        "img_size": [W, H],
        "board_bbox": [round(v, 1) for v in best["bbox"]],
        "board_area_px": round(best["area_px"], 1),
        "crop_box": list(crop_box),
        "abs_to_rel": abs_to_rel,
        "gt_centerlines_abs": gt_centerlines_abs,
        "gt_centerlines_rel": gt_centerlines_rel,
        "n_total_vectors": len(vectors),
        "n_centerlines": len(abs_to_rel),
        "panel": panel,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    scenes = sorted([p for p in ZEROSHOT_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(scenes)} zeroshot scenes")
    for scene in scenes:
        out_dir = OUT_DIR / scene.name
        meta = process_scene(scene, out_dir)
        if "error" in meta:
            print(f"  ✗ {scene.name}: {meta['error']}")
            continue
        print(f"  ✓ {scene.name}")
        print(f"      rep_ts={meta['rep_ts']}  board_area={meta['board_area_px']:.0f}px²")
        print(f"      centerlines: {meta['n_centerlines']}/{meta['n_total_vectors']} vectors")
        print(f"      abs→rel: {meta['abs_to_rel']}")
        print(f"      GT (rel idx): {meta['gt_centerlines_rel']}")
        print(f"      → {out_dir}/visual_prompt.jpg")


if __name__ == "__main__":
    main()
