"""SignVLM per-frame inference for trajectory demo.

For each frame in a scene, build sample → run model → parse target_lane.
Dumps a jsonl: one row per (scene, frame) with the resolved lane vec_id.

Reuses dataset.py sample builders. Sample construction at non-best frames:
  - cropped_sign: kept from best_board_frame (closest to training distribution
    — sign size/angle/lighting in that frame is what the model saw)
  - visual_prompt: rendered using the *current* frame's image + ego pose, so
    lane projections move with the ego. abs_to_rel mapping is scene-level
    (same per scene) to keep parse compatible.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_signvlm_infer.py \
      --adapter runs/sft/20260513_173317_pid3022191/final \
      --scene 00002ed8da8843a680469cf65afecddd \
      --out experiments/trajectory_smoke/00002ed8_C03_lanes.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dataset import _expand_bbox, ATTR_KEYS  # noqa: E402
from projection import (  # noqa: E402
    best_board_frame, centerline_relative_indices,
    panel_from_attr_info, render_visual_prompt,
)
from prompt import PROMPT_TEMPLATE  # noqa: E402
from metric import parse_response  # noqa: E402

SYSTEM_PROMPT = (
    "You are a sign-aware autonomous-driving assistant. "
    "Read the traffic sign and the rendered front view, then output a JSON dict "
    "with rules / lane_assignment / plan as instructed."
)


def _resize_for_qwen(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BICUBIC)


def build_sample_at_frame(scene_data: dict, scene_dir: Path,
                          frame_ts: int, sign_ts: int,
                          sign_bbox: tuple,
                          abs_to_rel: dict,
                          panel: dict):
    """Build (cropped_sign, visual_prompt) PIL images for inference at given frame.

    scene_data: pre-loaded dict with K, poses, vectors, etc.
    frame_ts:   the frame to render visual_prompt at
    sign_ts:    the frame from which cropped_sign is taken (usually best_board)
    sign_bbox:  bbox of board in sign_ts image (x0,y0,x1,y1)
    abs_to_rel: scene-level vector id → relative index
    panel:      pre-built metadata panel dict (panel_from_attr_info)
    """
    K = scene_data["K"]
    poses = scene_data["poses"]
    vectors = scene_data["vectors"]
    quat_order = "xyzw"
    pose_convention = "cam_to_world"

    sign_img = cv2.imread(str(scene_dir / "img" / f"{sign_ts}.jpg"))
    if sign_img is None:
        raise FileNotFoundError(f"sign img missing: {sign_ts}.jpg")
    H, W = sign_img.shape[:2]
    cx0, cy0, cx1, cy1 = _expand_bbox(sign_bbox, 1.4, 1.6, (W, H))
    cropped_bgr = sign_img[cy0:cy1, cx0:cx1]

    frame_img = cv2.imread(str(scene_dir / "img" / f"{frame_ts}.jpg"))
    if frame_img is None:
        raise FileNotFoundError(f"frame img missing: {frame_ts}.jpg")

    vp_bgr, _ = render_visual_prompt(
        frame_img, vectors, K, poses[str(frame_ts)],
        quat_order, pose_convention,
        abs_to_rel=abs_to_rel,
        metadata_panel=panel,
    )

    return (
        Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)),
        Image.fromarray(cv2.cvtColor(vp_bgr, cv2.COLOR_BGR2RGB)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="path to LoRA adapter dir (with adapter_config.json)")
    ap.add_argument("--scene", required=True, help="scene hash")
    ap.add_argument("--mapdr-root",
                    default=str(Path(__file__).resolve().parent.parent / "data" / "MapDR"))
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--model-name", default="",
                    help="if empty, auto-detect from <adapter>/../args.json")
    ap.add_argument("--max-image-pixels", type=int, default=0,
                    help="if 0, auto-detect")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--perturb-mode", choices=("none", "conflict"), default="none",
                    help="If 'conflict', perturb panel field at each frame (deterministic per scene)")
    ap.add_argument("--perturb-seed", type=int, default=42)
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="if >0, only process the first N frames (smoke test)")
    args = ap.parse_args()

    # Auto-detect model_name and max_pixels from args.json
    adapter_dir = Path(args.adapter).resolve()
    args_json = adapter_dir.parent / "args.json"
    if args_json.exists():
        a = json.loads(args_json.read_text())
        if not args.model_name:
            args.model_name = a.get("model_name", "Qwen2.5-VL-3B-Instruct")
        if args.max_image_pixels == 0:
            args.max_image_pixels = int(a.get("max_image_pixels", 802816))
    else:
        if not args.model_name:
            args.model_name = "Qwen2.5-VL-3B-Instruct"
        if args.max_image_pixels == 0:
            args.max_image_pixels = 802816

    model_dir = REPO / "ckpts" / args.model_name
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")

    scene_dir = Path(args.mapdr_root) / args.scene
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[infer] adapter = {adapter_dir}")
    print(f"[infer] model = {args.model_name}  max_pixels = {args.max_image_pixels}")
    print(f"[infer] scene = {args.scene}")
    print(f"[infer] out = {out_path}")

    # ---------- load model ----------
    t0 = time.time()
    print(f"[infer] loading processor + base + adapter…")
    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.to("cuda:0").eval()
    print(f"[infer] load done in {time.time() - t0:.1f}s")

    # ---------- precompute scene-level fixtures ----------
    data = json.loads((scene_dir / "data.json").read_text())
    label = json.loads((scene_dir / "label.json").read_text())
    K = np.array(data["camera_intrinsic_matrix"])
    poses = data["camera_pose"]
    board = np.array(data["traffic_board_pose"])
    vectors = data["vector"]
    scene_data = {"K": K, "poses": poses, "vectors": vectors}

    # Best board frame → fixed cropped_sign source
    first_img = cv2.imread(str(scene_dir / "img" / f"{sorted(poses.keys(), key=int)[0]}.jpg"))
    H, W = first_img.shape[:2]
    best_ts, best = best_board_frame(board, K, poses, (W, H), "xyzw", "cam_to_world")
    if best_ts is None or best is None:
        raise SystemExit(f"no valid best_board_frame for {args.scene}")
    print(f"[infer] best_board_frame ts = {best_ts}  bbox = {best['bbox']}")

    # abs_to_rel (scene-level, no shuffle)
    ordered_ids, abs_to_rel = centerline_relative_indices(vectors)
    rel_to_abs = {v: k for k, v in abs_to_rel.items()}

    # Metadata panel: clean OR conflict-perturbed (deterministic per scene+seed)
    rules_attr = [r.get("attr_info", {}) for r in label.values()]
    panel = panel_from_attr_info(rules_attr)
    perturb_info = None
    if args.perturb_mode == "conflict":
        import random as _random
        from perturb_conflict import apply_conflict_to_panel
        rng = _random.Random(args.perturb_seed + hash(args.scene) % 10000)
        panel, perturb_info = apply_conflict_to_panel(panel, rng=rng)
        print(f"[perturb] conflict applied: type={perturb_info['type']} "
              f"field={perturb_info['field']} '{perturb_info['original']}' → '{perturb_info['perturbed']}'")

    # Frame list
    ts_sorted = sorted([int(t) for t in poses.keys()])
    if args.limit_frames > 0:
        ts_sorted = ts_sorted[:args.limit_frames]
    print(f"[infer] {len(ts_sorted)} frames to process")

    # ---------- per-frame inference ----------
    out_f = open(out_path, "w")
    n_parse_fail = 0
    t_loop = time.time()
    for fi, ts in enumerate(ts_sorted):
        t_f = time.time()
        try:
            sign_img, vp_img = build_sample_at_frame(
                scene_data, scene_dir, ts, best_ts, best["bbox"],
                abs_to_rel, panel,
            )
        except Exception as e:
            print(f"[frame {fi}] sample build failed: {e}")
            continue
        sign_img = _resize_for_qwen(sign_img, args.max_image_pixels)
        vp_img = _resize_for_qwen(vp_img, args.max_image_pixels)

        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",
             "content": [
                 {"type": "image", "image": sign_img},
                 {"type": "image", "image": vp_img},
                 {"type": "text", "text": PROMPT_TEMPLATE},
             ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[[sign_img, vp_img]],
            padding=True, return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        new_ids = out_ids[:, inputs.input_ids.shape[1]:]
        pred_text = processor.batch_decode(
            new_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        parsed = parse_response(pred_text)
        parse_ok = parsed is not None
        target_lane_rel = -1
        target_lane_vec_id = None
        if parse_ok:
            plan = parsed.get("plan", {}) or {}
            target_lane_rel = plan.get("target_lane", -1)
            if isinstance(target_lane_rel, int) and target_lane_rel in rel_to_abs:
                target_lane_vec_id = rel_to_abs[target_lane_rel]
        else:
            n_parse_fail += 1

        rec = {
            "scene_id": args.scene,
            "frame_idx": fi,
            "ts_ns": ts,
            "is_best_board_frame": (ts == best_ts),
            "target_lane_rel": target_lane_rel,
            "target_lane_vec_id": target_lane_vec_id,
            "parse_ok": parse_ok,
            "latency_s": round(time.time() - t_f, 2),
            "pred_text": pred_text,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

        cum = time.time() - t_loop
        avg = cum / (fi + 1)
        print(f"[frame {fi:02d}/{len(ts_sorted)-1}  ts={ts}  "
              f"lane_rel={target_lane_rel} → vec_id={target_lane_vec_id}  "
              f"parse={'✓' if parse_ok else '✗'}  "
              f"dt={rec['latency_s']:.1f}s  avg={avg:.1f}s/frame]")

    out_f.close()
    total = time.time() - t_loop
    print(f"\n[infer] done. {len(ts_sorted)} frames in {total:.0f}s "
          f"({total/max(1,len(ts_sorted)):.1f}s/frame)  parse_fail = {n_parse_fail}")
    print(f"[infer] output → {out_path}")


if __name__ == "__main__":
    main()
