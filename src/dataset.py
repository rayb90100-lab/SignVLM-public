"""Scene-level MapDR dataset for sign-aware VLA training (路线 3: discrete VLA).

For each scene, produces:
  - cropped_sign:  PIL.Image (Picture 1)
  - visual_prompt: PIL.Image (Picture 2 — centerlines + relative-index labels)
  - prompt_text:   str (RuleVLM-style synonym + plan instruction)
  - gt_text:       str (Python str(dict): rules + lane_assignment + plan, all discrete)
  - meta:          dict (scene_id / rep_ts / abs_to_rel / future_pose_local /
                        future_pose_enu / perturbed flags) — meta carries the
                  trajectory data that 路线 1 (AutoVLA-style) will tokenize.

Path 路线 3 → 路线 1 upgrade:
  - dataloader main body, geometry, double-image rendering, shuffle, map
    perturbation are all reused (~85% code reuse).
  - To upgrade: add an ActionTokenizer + codebook (clustered from MapDR train
    trajectories), then transform meta['future_pose_local'] into <action_N>
    sequences and append to the plan field of gt_text.
"""
from __future__ import annotations
import copy
import json
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from projection import (  # noqa: E402
    best_board_frame,
    centerline_relative_indices,
    panel_from_attr_info,
    quat_to_rot,
    render_visual_prompt,
)
from prompt import PROMPT_TEMPLATE  # noqa: E402
from perturb_conflict import apply_conflict_to_panel, apply_multi_field_conflict_to_panel  # noqa: E402


# Standard order/keys we use in attr_info — sorted alphabetically (matches RuleVLM evaluate.py)
ATTR_KEYS = ("AllowedTransport", "EffectiveDate", "EffectiveTime",
             "HighSpeedLimit", "LaneDirection", "LaneType",
             "LowSpeedLimit", "RuleIndex")


def _expand_bbox(bbox, ex, ey, img_size):
    W, H = img_size
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = (x1 - x0) * ex
    h = (y1 - y0) * ey
    return (max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
            min(W, int(cx + w / 2)), min(H, int(cy + h / 2)))


class MapDRDataset(Dataset):
    """MapDR scenes → (cropped_sign, visual_prompt, prompt_text, gt_text, meta).

    Parameters
    ----------
    root : path to mapdr_v1220 (or symlink). Each subdir is one scene.
    split : 'Train' or 'Test' (key in split.json)
    split_json : path to split.json. Default: root.parent / 'split.json'.
    shuffle_idx : training-time de-bias of relative centerline indices
    perturb_mode : 'none' | 'noise' | 'conflict'
        - 'noise'    = legacy `_perturb_map`: changes label+input together
                       (Task 6.9 ablation; input-fidelity training, not robustness)
        - 'conflict' = CAVP vision-map conflict: only the rendered map panel
                       value is perturbed; label / GT untouched. Main result of
                       Novelty 2. See docs/CONFLICT_PERTURB_DESIGN.md
    perturb_prob : per-sample probability of applying perturbation
    conflict_types : restrict 'conflict' mode to a subset of
        {'speed','direction','vehicle','time'}. Default = all 4.
    map_perturb_prob : DEPRECATED. Kept for backward compat — equivalent to
        `perturb_mode='noise'` + `perturb_prob=<value>`.
    plan_n_frames : how many evenly-spaced future frames to capture in meta
    plan_horizon_s : capture window after the representative frame
    quat_order, pose_convention : forwarded to projection helpers
    """

    def __init__(self,
                 root: str | Path,
                 split: str = "Train",
                 split_json: Optional[str | Path] = None,
                 shuffle_idx: bool = True,
                 perturb_mode: str = "none",
                 perturb_prob: float = 0.0,
                 conflict_types: Optional[list] = None,
                 conflict_type_weights: Optional[list] = None,
                 n_conflict_fields: int = 1,  # multi-field ablation eval (1 = single, 2-4 = multi-field)
                 map_perturb_prob: float = 0.0,
                 plan_n_frames: int = 8,
                 plan_horizon_s: float = 3.0,
                 quat_order: str = "xyzw",
                 pose_convention: str = "cam_to_world",
                 seed: int = 42):
        self.root = Path(root).resolve()
        self.split = split
        self.shuffle_idx = shuffle_idx

        # Back-compat: old API → new perturb_mode='noise'
        if map_perturb_prob > 0 and perturb_mode == "none":
            perturb_mode = "noise"
            perturb_prob = map_perturb_prob
        if perturb_mode not in ("none", "noise", "conflict"):
            raise ValueError(
                f"perturb_mode must be 'none' / 'noise' / 'conflict'; "
                f"got {perturb_mode!r}"
            )
        self.perturb_mode = perturb_mode
        self.perturb_prob = float(perturb_prob)
        self.conflict_types = list(conflict_types) if conflict_types else \
            ["speed", "direction", "vehicle", "time"]
        if conflict_type_weights is not None:
            if len(conflict_type_weights) != len(self.conflict_types):
                raise ValueError(
                    f"conflict_type_weights length {len(conflict_type_weights)} "
                    f"!= conflict_types length {len(self.conflict_types)}")
            self.conflict_type_weights = [float(w) for w in conflict_type_weights]
        else:
            self.conflict_type_weights = None
        self.n_conflict_fields = int(n_conflict_fields)
        if self.n_conflict_fields < 1 or self.n_conflict_fields > 4:
            raise ValueError(f"n_conflict_fields {n_conflict_fields} must be in [1, 4]")
        # Legacy attribute name kept for external readers
        self.map_perturb_prob = self.perturb_prob if perturb_mode == "noise" else 0.0

        self.plan_n_frames = plan_n_frames
        self.plan_horizon_s = plan_horizon_s
        self.quat_order = quat_order
        self.pose_convention = pose_convention

        if split_json is None:
            split_json = self.root.parent / "split.json"
        splits = json.loads(Path(split_json).read_text())
        if split not in splits:
            raise ValueError(f"split must be one of {list(splits.keys())}; got {split!r}")
        self.scene_ids = list(splits[split])
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.scene_ids[idx]
        scene_dir = self.root / sid
        data = json.loads((scene_dir / "data.json").read_text())
        label = json.loads((scene_dir / "label.json").read_text())

        # Optional HD-Map perturbation BEFORE rendering / GT building.
        # 'noise' mode mutates label+data (input-fidelity training); 'conflict'
        # mode leaves them alone — only the rendered panel will be perturbed
        # later (vision-faithful training, GT永远按视觉).
        perturbed = False
        conflict_meta = None
        do_perturb = (
            self.perturb_mode != "none"
            and self.perturb_prob > 0
            and self.rng.random() < self.perturb_prob
        )
        if do_perturb and self.perturb_mode == "noise":
            label, data = self._perturb_map(label, data)
            perturbed = True

        K = np.array(data["camera_intrinsic_matrix"])
        poses = data["camera_pose"]
        board = np.array(data["traffic_board_pose"])
        vectors = data["vector"]

        img_files = sorted((scene_dir / "img").glob("*.jpg"))
        if not img_files:
            raise FileNotFoundError(f"no images in {scene_dir}")
        first_img = cv2.imread(str(img_files[0]))
        H, W = first_img.shape[:2]

        best_ts, best = best_board_frame(
            board, K, poses, (W, H), self.quat_order, self.pose_convention
        )
        if best_ts is None:
            # Fallback: pick first frame; mark in meta. Caller can filter.
            best_ts = sorted(poses.keys(), key=int)[0]
            best = None

        rep_img = cv2.imread(str(scene_dir / "img" / f"{best_ts}.jpg"))
        if rep_img is None:
            raise FileNotFoundError(f"can't read img {best_ts}.jpg in {scene_dir}")

        # Picture 1: cropped sign
        if best is not None:
            cx0, cy0, cx1, cy1 = _expand_bbox(best["bbox"], 1.4, 1.6, (W, H))
            cropped_bgr = rep_img[cy0:cy1, cx0:cx1]
        else:
            cropped_bgr = rep_img

        # Build relative-index mapping (with optional shuffle)
        ordered_ids, abs_to_rel = centerline_relative_indices(vectors)
        if self.shuffle_idx and len(ordered_ids) > 1:
            shuffled = list(ordered_ids)
            self.rng.shuffle(shuffled)
            abs_to_rel = {vid: i for i, vid in enumerate(shuffled)}

        # Build metadata panel (always rendered, even in clean training, so
        # train/eval distribution stays consistent). For conflict mode, the
        # rendered panel is perturbed but label/GT remain visual-truth.
        rules_attr = [r.get("attr_info", {}) for r in label.values()]
        panel_clean = panel_from_attr_info(rules_attr)
        if do_perturb and self.perturb_mode == "conflict":
            if self.n_conflict_fields == 1:
                panel_to_render, conflict_meta = apply_conflict_to_panel(
                    panel_clean, rng=self.rng, types_pool=self.conflict_types,
                    type_weights=self.conflict_type_weights,
                )
            else:
                panel_to_render, conflict_meta = apply_multi_field_conflict_to_panel(
                    panel_clean, n_fields=self.n_conflict_fields,
                    rng=self.rng, types_pool=self.conflict_types,
                )
            perturbed = True
        else:
            panel_to_render = panel_clean

        # Picture 2: visual prompt with caller-supplied abs_to_rel
        vp_bgr, _ = render_visual_prompt(
            rep_img, vectors, K, poses[best_ts],
            self.quat_order, self.pose_convention,
            abs_to_rel=abs_to_rel,
            metadata_panel=panel_to_render,
        )
        if best is not None:
            bx0, by0, bx1, by1 = (int(v) for v in best["bbox"])
            cv2.rectangle(vp_bgr, (bx0, by0), (bx1, by1), (0, 255, 0), 3)

        # GT construction (rules + plan)
        rules_gt = self._build_rules_gt(label, abs_to_rel)
        plan_gt = self._build_plan_gt(label, abs_to_rel, poses, best_ts)
        gt_dict = {"rules": rules_gt, "plan": plan_gt}
        gt_text = str(gt_dict)  # Python str(dict): single-quoted (RuleVLM evaluate.py converts ' → " before json.loads)

        # Future-pose data for meta — used by 路线 1 (AutoVLA-style) trajectory tokenization
        future_pose_local, future_pose_enu = self._extract_future_pose(poses, best_ts)

        return {
            "cropped_sign": Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)),
            "visual_prompt": Image.fromarray(cv2.cvtColor(vp_bgr, cv2.COLOR_BGR2RGB)),
            "prompt_text": PROMPT_TEMPLATE,
            "gt_text": gt_text,
            "meta": {
                "scene_id": sid,
                "rep_ts": best_ts,
                "abs_to_rel": abs_to_rel,
                "future_pose_local": future_pose_local.tolist() if future_pose_local is not None else [],
                "future_pose_enu": future_pose_enu.tolist() if future_pose_enu is not None else [],
                "perturbed": perturbed,
                "perturb_mode": self.perturb_mode,
                "conflict_meta": conflict_meta,
                "panel_clean": panel_clean,
                "panel_rendered": panel_to_render,
                "n_centerlines": len(abs_to_rel),
                "n_rules": len(rules_gt),
            },
        }

    # -- GT builders ---------------------------------------------------------

    @staticmethod
    def _build_rules_gt(label: dict, abs_to_rel: dict) -> list:
        """Convert label.json rules to ordered list with attr_info sorted &
        centerline remapped to relative indices (sorted)."""
        out = []
        for rid in sorted(label.keys(), key=lambda x: int(x) if x.lstrip("-").isdigit() else x):
            r = label[rid]
            attr = r.get("attr_info", {})
            attr_sorted = {k: attr.get(k, "None") for k in ATTR_KEYS}
            cl_rel = sorted(
                abs_to_rel[str(c)]
                for c in r.get("centerline", [])
                if str(c) in abs_to_rel
            )
            out.append({"attr_info": attr_sorted, "centerline": cl_rel})
        return out

    @staticmethod
    def _build_plan_gt(label: dict, abs_to_rel: dict, poses: dict, rep_ts: str) -> dict:
        """Heuristic plan GT (路线 3 minimum):
          - target_lane: first rule's first centerline (relative index)
          - action: 'stay' placeholder (路线 1 升级时从 future trajectory lateral
                   offset 推断)
          - reasoning: empty string (CoT 训练时由模型生成)
        """
        target_lane = -1
        for rid in sorted(label.keys(), key=lambda x: int(x) if x.lstrip("-").isdigit() else x):
            cl = label[rid].get("centerline", [])
            for c in cl:
                if str(c) in abs_to_rel:
                    target_lane = abs_to_rel[str(c)]
                    break
            if target_lane != -1:
                break
        return {
            "target_lane": target_lane,
            "action": "stay",
            "reasoning": "",
        }

    # -- Future pose extraction ---------------------------------------------

    def _extract_future_pose(self, poses: dict, rep_ts: str):
        """Sample plan_n_frames evenly within (rep_ts, rep_ts + plan_horizon_s].

        Returns
        -------
        local : (n, 3) np.ndarray of self-frame coordinates relative to the
                ego pose at rep_ts. None if rep_ts not found or no future frames.
        enu   : (n, 3) np.ndarray of absolute ENU positions for the same frames.
        """
        ts_sorted = sorted(poses.keys(), key=int)
        if rep_ts not in poses:
            return None, None
        rep_idx = ts_sorted.index(rep_ts)
        rep_t_ns = int(rep_ts)
        horizon_ns = self.plan_horizon_s * 1e9
        future_ts = [t for t in ts_sorted[rep_idx + 1:]
                     if (int(t) - rep_t_ns) <= horizon_ns]
        if not future_ts:
            return None, None
        n = min(self.plan_n_frames, len(future_ts))
        idxs = np.linspace(0, len(future_ts) - 1, num=n).astype(int)
        sampled_ts = [future_ts[i] for i in idxs]
        enu = np.array([poses[t]["tvec_enu"] for t in sampled_ts], dtype=np.float64)

        # Local frame: subtract rep ENU then rotate by inverse rep heading.
        rep_pose = poses[rep_ts]
        rep_t = np.asarray(rep_pose["tvec_enu"], dtype=np.float64)
        R_rep = quat_to_rot(rep_pose["rvec_enu"], self.quat_order)
        # cam_to_world: P_cam = R^T (P_world - t). For pose representation,
        # the body-frame relative position of a future ENU point is the same.
        local = (enu - rep_t) @ R_rep
        return local, enu

    # -- HD-Map perturbation ------------------------------------------------

    def _perturb_map(self, label: dict, data: dict):
        """Inject one HD-Map noise per sample (deepcopy in-place safe).

        Three classes (uniform): direction flip / speed change / centerline swap.
        Increase variety in path 路线 1 升级时 if needed.
        """
        label = copy.deepcopy(label)
        data = copy.deepcopy(data)
        roll = self.rng.random()
        if roll < 1 / 3:
            flip = {"TurnLeft": "TurnRight", "TurnRight": "TurnLeft",
                    "GoStraight": "TurnLeft", "UTurn": "TurnRight"}
            for r in label.values():
                ld = r.get("attr_info", {}).get("LaneDirection", [])
                if isinstance(ld, list) and ld:
                    r["attr_info"]["LaneDirection"] = [flip.get(d, d) for d in ld]
                    break
        elif roll < 2 / 3:
            for r in label.values():
                attr = r.get("attr_info", {})
                v = attr.get("HighSpeedLimit", "None")
                if v != "None":
                    try:
                        attr["HighSpeedLimit"] = str(max(20, int(v) - 20))
                    except (ValueError, TypeError):
                        pass
                    break
        else:
            cl_ids = [k for k, v in data["vector"].items() if str(v.get("type")) == "3"]
            if len(cl_ids) >= 2:
                a, b = self.rng.sample(cl_ids, 2)
                data["vector"][a], data["vector"][b] = data["vector"][b], data["vector"][a]
        return label, data
