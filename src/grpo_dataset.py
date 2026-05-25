"""GRPO dataset for SignVLM conflict-resolution training.

Unlike DPO (which needs explicit chosen/rejected pairs), GRPO only needs
the prompt + ground-truth meta — the model self-samples G rollouts and
the reward function (src/grpo_reward.py) scores each against GT.

For each scene, produces:
  - cropped_sign:    PIL.Image (visual ground truth — never perturbed)
  - visual_prompt:   PIL.Image (panel ALWAYS conflict-perturbed, since
                                 GRPO reward only meaningful on conflict scenes)
  - prompt_text:     str (same template as SFT/DPO)
  - gt_text:         str (vision-faithful answer, used by reward fn)
  - conflict_meta:   dict (type / panel_value / gt_value — passed to reward)
  - scene_id:        str (for logging / debugging)

Note: GRPO trains exclusively on conflict scenes (perturb_prob=1.0).
This is intentional — the conflict_reward signal is undefined on
non-conflict scenes (would always return 0.0, no learning signal).
If we want a "no-perturb regularization" pass, do it via the KL anchor
(SFT-init ref adapter) instead of mixing non-conflict samples in batch.

See docs/RFT_GRPO_DESIGN.md §2.5.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from dataset import MapDRDataset  # noqa: E402


class MapDRGRPODataset(Dataset):
    """Thin wrap over MapDRDataset(perturb_mode='conflict') for GRPO.

    Args:
        root: MapDR dataset root (containing data.json / label.json / img/)
        split: 'Train' | 'Test'
        shuffle_idx: bool — shuffle centerline relative indices (training: True)
        conflict_types: optional list of allowed conflict types
                        (e.g. ['speed', 'direction'])
        conflict_type_weights: optional dict[str, float] weights for each
                                conflict type (defaults to uniform)
        seed: random seed for perturb sampling
    """

    def __init__(self, root: Path, split: str = "Train",
                  shuffle_idx: bool = True,
                  conflict_types: Optional[list[str]] = None,
                  conflict_type_weights: Optional[dict[str, float]] = None,
                  seed: int = 42) -> None:
        self.inner = MapDRDataset(
            root=root,
            split=split,
            shuffle_idx=shuffle_idx,
            perturb_mode="conflict",
            perturb_prob=1.0,         # every scene perturbed (GRPO core assumption)
            conflict_types=conflict_types,
            conflict_type_weights=conflict_type_weights,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        s = self.inner[idx]
        # MapDRDataset returns: cropped_sign, visual_prompt, prompt_text, gt_text, meta
        meta = s.get("meta", {}) or {}
        return {
            "cropped_sign": s["cropped_sign"],
            "visual_prompt": s["visual_prompt"],
            "prompt_text": s["prompt_text"],
            "gt_text": s["gt_text"],
            "conflict_meta": meta.get("conflict_meta"),
            "scene_id": meta.get("scene_id", f"idx_{idx}"),
        }


__all__ = ["MapDRGRPODataset"]
