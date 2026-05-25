"""DPO pair dataset for vision-faithful conflict resolution.

For each scene, produces:
  - cropped_sign:  PIL.Image (visual ground truth — never perturbed)
  - visual_prompt: PIL.Image (panel may be conflict-perturbed)
  - prompt_text:   str (same as SFT)
  - chosen_text:   str (= MapDRDataset gt_text, vision-faithful answer)
  - rejected_text: str (chosen with panel-trust fields overwritten)
  - meta:          dict (pair_type='conflict'|'control', conflict_meta, ...)

Two pair classes mixed by `conflict_ratio` (default 0.7):
  - conflict pair: visual_prompt shows perturbed panel → rejected = chosen
    with the conflicted field overridden to match the panel. Trains the
    model to prefer the vision-faithful answer over the panel-trust one
    when the panel lies.
  - control pair : visual_prompt shows the clean (truthful) panel →
    rejected = chosen with a NON-conflict field perturbed (LaneType
    flipped). Prevents trivial "always ignore panel" overfit by making
    sure the model still learns that off-panel answers can also be wrong.

See docs/RFT_DPO_DESIGN.md §2 for design rationale.
"""
from __future__ import annotations
import ast
import copy
import random
import sys
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from dataset import MapDRDataset  # noqa: E402
from perturb_conflict import panel_to_rejected_fields  # noqa: E402


# Control-pair: flip LaneType to a different valid value. LaneType is a
# scene-stable string in attr_info; flipping it generates a clearly wrong
# rejected answer that does NOT overlap with panel-trust mode behavior.
_LANE_TYPE_POOL = (
    "DirectionLane", "MultiLane", "BusLane", "EmergencyLane",
    "TidalLane", "VariableDirLane", "SpeedLimitedLane",
)


def _override_attr_fields(gt_dict: dict, overrides: dict) -> dict:
    """Return a deep copy of gt_dict with `overrides` applied to every rule's
    attr_info. Values in `overrides` replace existing entries; missing keys
    are added.
    """
    new = copy.deepcopy(gt_dict)
    for rule in new.get("rules", []):
        attr = rule.setdefault("attr_info", {})
        for k, v in overrides.items():
            attr[k] = v
    return new


def _control_overrides(gt_dict: dict, rng: random.Random) -> dict:
    """Pick a different LaneType than the one(s) currently in the GT."""
    current = {
        rule.get("attr_info", {}).get("LaneType")
        for rule in gt_dict.get("rules", [])
    }
    candidates = [t for t in _LANE_TYPE_POOL if t not in current]
    if not candidates:
        candidates = list(_LANE_TYPE_POOL)
    return {"LaneType": rng.choice(candidates)}


class MapDRDPODataset(Dataset):
    """Wraps two MapDRDataset views (conflict / clean) and emits DPO pairs.

    Parameters
    ----------
    root, split, split_json : same semantics as MapDRDataset
    conflict_ratio : float in [0, 1]; probability that a given index emits
        a conflict pair (vs control pair). Default 0.7.
    shuffle_idx : forwarded to both underlying datasets
    conflict_types, conflict_type_weights : restrict conflict pair
        generation to a subset / weight types (e.g. 0.15/0.55/0.15/0.15 to
        bias direction). Default uniform over all 4 classes.
    seed : base seed; the two underlying MapDRDataset use seed and seed+1
        to keep their internal rngs decorrelated.
    Other kwargs are forwarded to MapDRDataset (plan_*, quat_order, etc.)
    """

    def __init__(self,
                 root: str | Path,
                 split: str = "Train",
                 split_json: Optional[str | Path] = None,
                 conflict_ratio: float = 0.7,
                 shuffle_idx: bool = True,
                 conflict_types: Optional[list] = None,
                 conflict_type_weights: Optional[list] = None,
                 seed: int = 42,
                 **kwargs):
        if not 0.0 <= conflict_ratio <= 1.0:
            raise ValueError(f"conflict_ratio must be in [0,1]; got {conflict_ratio}")
        self.conflict_ratio = conflict_ratio
        self.rng = random.Random(seed + 2026)  # mix-decision rng — independent of dataset rngs

        common = dict(
            root=root, split=split, split_json=split_json,
            shuffle_idx=shuffle_idx, **kwargs,
        )
        # View A: every sample is conflict-perturbed (panel lies)
        self.ds_conflict = MapDRDataset(
            perturb_mode="conflict", perturb_prob=1.0,
            conflict_types=conflict_types,
            conflict_type_weights=conflict_type_weights,
            seed=seed,
            **common,
        )
        # View B: clean panel (panel == GT)
        self.ds_clean = MapDRDataset(
            perturb_mode="none", perturb_prob=0.0,
            seed=seed + 1,
            **common,
        )
        # Both views walk the same scene_ids list, so __len__ aligns.
        assert len(self.ds_conflict) == len(self.ds_clean)

    def __len__(self) -> int:
        return len(self.ds_conflict)

    def __getitem__(self, idx: int) -> dict:
        is_conflict = self.rng.random() < self.conflict_ratio

        if is_conflict:
            sample = self.ds_conflict[idx]
            cmeta = sample["meta"].get("conflict_meta")
            if cmeta is None:
                # Shouldn't happen since perturb_prob=1.0, but be defensive —
                # fall back to control pair if conflict didn't materialize.
                is_conflict = False

        if not is_conflict:
            sample = self.ds_clean[idx]
            cmeta = None

        # Parse chosen back to dict (str(dict) → dict, safely)
        chosen_dict = ast.literal_eval(sample["gt_text"])

        if is_conflict:
            overrides = panel_to_rejected_fields(cmeta)
            pair_type = "conflict"
        else:
            overrides = _control_overrides(chosen_dict, self.rng)
            pair_type = "control"

        rejected_dict = _override_attr_fields(chosen_dict, overrides)
        rejected_text = str(rejected_dict)

        meta = dict(sample["meta"])
        meta["pair_type"] = pair_type
        meta["rejected_overrides"] = overrides

        return {
            "cropped_sign":  sample["cropped_sign"],
            "visual_prompt": sample["visual_prompt"],
            "prompt_text":   sample["prompt_text"],
            "chosen_text":   sample["gt_text"],
            "rejected_text": rejected_text,
            "meta":          meta,
        }
