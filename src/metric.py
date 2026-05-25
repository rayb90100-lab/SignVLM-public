"""SignVLM evaluation metrics — clean-room reimplementation.

Reimplements the 5 RuleVLM/MapDR metrics from their published formulas
(see the MapDR/RuleVLM paper). This file works purely from the formulas;
no source-code copy from the official RuleVLM repo.

Metrics
-------
Scene-level 0/1:
  - Understand_Acc : 1 iff the multiset of attr_info dicts is identical
  - Asso_Acc       : 1 iff the multiset of centerline lists is identical
  - ALL_Acc        : Understand_Acc AND Asso_Acc

Rule-level micro (multiset 1-on-1 match across the entire eval set):
  - RE_P / RE_R / RE_F1 : tp = pred attr_info found in (remaining) gt attr_info
                          multiset; P = tp/pred_total, R = tp/gt_total

(rule, single_centerline) pair-level micro:
  - Overall_P / Overall_R / Overall_F1 : same idea but on (attr_info, single_cl) pairs

Plan field extensions (SignVLM-specific, not in RuleVLM):
  - lane_correctness : pred.plan.target_lane == gt.plan.target_lane
                       computed only over scenes whose gt provides a target_lane

Parse failures
--------------
- text.replace("'", '"') → json.loads (RuleVLM evaluate.py trick)
- on parse failure: pred contributes nothing (tp=fp=0), but gt counts still
  enter the denominator → R falls, P unaffected. Scene flags all False.
"""
from __future__ import annotations
import json
import re
from typing import Any, Optional


# -- Parsing --------------------------------------------------------------

def parse_response(text: str) -> Optional[dict]:
    """Parse a Python str(dict)-style response into a dict, or None on failure."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    try:
        return json.loads(s.replace("'", '"'))
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0).replace("'", '"'))
            except json.JSONDecodeError:
                return None
        return None


def sort_lists_in_dict(obj: Any) -> Any:
    """Recursively sort all lists so that {centerline: [1,2]} == {centerline: [2,1]}."""
    if isinstance(obj, dict):
        return {k: sort_lists_in_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [sort_lists_in_dict(x) for x in obj]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    return obj


def _attr_key(attr: dict) -> str:
    """Hashable canonical form of an attr_info dict."""
    return json.dumps(attr, sort_keys=True, ensure_ascii=False)


def _multiset_intersect(pred: list, gt: list) -> int:
    """Count items matched 1-on-1 across two multisets (each gt slot used at most once)."""
    remaining = list(gt)
    tp = 0
    for p in pred:
        if p in remaining:
            remaining.remove(p)
            tp += 1
    return tp


# -- Metric accumulator ---------------------------------------------------

class MapDRMetric:
    """Accumulate over scenes; call `.compute()` to get final ratios."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_scenes = 0
        self.understand_correct = 0
        self.asso_correct = 0
        self.both_correct = 0
        self.parse_failures = 0
        # R.E. (rule-level) micro
        self.re_tp = 0
        self.re_pred = 0  # tp + fp
        self.gt_rules = 0  # R denominator
        # Overall ((rule, single_centerline) pair) micro
        self.ov_tp = 0
        self.ov_pred = 0
        self.gt_pairs = 0
        # plan extensions
        self.target_lane_correct = 0
        self.target_lane_present = 0  # gt has a real target_lane (≠ -1)
        # CAVP conflict tracking (only populated if scenes carry conflict_meta).
        # visual-faithful = scene's pred matches the (untouched) GT, which
        # mirrors the visual truth. Failure modes (parse-fail / understand-mis
        # / asso-mis) all count against visual-faithfulness.
        self.conflict_total: dict[str, int] = {}
        self.conflict_correct: dict[str, int] = {}

    def add(self, pred_text: str, gt_text: str,
            conflict_meta: Optional[dict] = None) -> None:
        """Accumulate one scene.

        `conflict_meta`, if provided (CAVP eval), bumps the per-type
        conflict counter. A scene counts as visual-faithful iff both
        Understand and Asso match the GT (which equals the visual truth
        even under conflict perturb).
        """
        self.total_scenes += 1
        gt = parse_response(gt_text)
        if gt is None:
            # GT should always parse; if it doesn't, treat scene as malformed and skip.
            self.total_scenes -= 1
            return
        gt = sort_lists_in_dict(gt)
        gt_rules = gt.get("rules", []) or []
        gt_attr_keys = [_attr_key(r.get("attr_info", {})) for r in gt_rules]
        gt_pairs = []
        for r in gt_rules:
            ak = _attr_key(r.get("attr_info", {}))
            for c in r.get("centerline", []) or []:
                gt_pairs.append((ak, c))
        self.gt_rules += len(gt_attr_keys)
        self.gt_pairs += len(gt_pairs)

        # Plan target_lane GT availability (count even if pred fails to parse)
        gt_plan = gt.get("plan", {}) or {}
        gt_tl = gt_plan.get("target_lane", -1)
        if gt_tl != -1:
            self.target_lane_present += 1

        # Conflict subset bookkeeping (denominator counts even on parse fail)
        ctype = conflict_meta.get("type", "unknown") if conflict_meta else None
        if ctype is not None:
            self.conflict_total[ctype] = self.conflict_total.get(ctype, 0) + 1

        pred = parse_response(pred_text)
        if pred is None:
            self.parse_failures += 1
            return  # pred contributes 0 tp/fp; scene flags all False (default)
        pred = sort_lists_in_dict(pred)
        pred_rules = pred.get("rules", []) or []
        pred_attr_keys = [_attr_key(r.get("attr_info", {})) for r in pred_rules]
        pred_pairs = []
        for r in pred_rules:
            ak = _attr_key(r.get("attr_info", {}))
            for c in r.get("centerline", []) or []:
                pred_pairs.append((ak, c))

        # R.E. micro
        self.re_tp += _multiset_intersect(pred_attr_keys, gt_attr_keys)
        self.re_pred += len(pred_attr_keys)
        # Overall micro
        self.ov_tp += _multiset_intersect(pred_pairs, gt_pairs)
        self.ov_pred += len(pred_pairs)

        # Scene-level flags
        understand_ok = sorted(pred_attr_keys) == sorted(gt_attr_keys)
        pred_cls = sorted(tuple(r.get("centerline", []) or []) for r in pred_rules)
        gt_cls = sorted(tuple(r.get("centerline", []) or []) for r in gt_rules)
        asso_ok = pred_cls == gt_cls
        if understand_ok:
            self.understand_correct += 1
        if asso_ok:
            self.asso_correct += 1
        if understand_ok and asso_ok:
            self.both_correct += 1

        # Plan target_lane match
        pred_plan = pred.get("plan", {}) or {}
        pred_tl = pred_plan.get("target_lane", -1)
        if gt_tl != -1 and pred_tl == gt_tl:
            self.target_lane_correct += 1

        # CAVP: scene is visual-faithful if it matches the GT (which always
        # mirrors visual truth, since conflict perturb only mutates the panel).
        if ctype is not None and understand_ok and asso_ok:
            self.conflict_correct[ctype] = self.conflict_correct.get(ctype, 0) + 1

    def compute(self) -> dict:
        n = max(1, self.total_scenes)
        re_p = self.re_tp / max(1, self.re_pred)
        re_r = self.re_tp / max(1, self.gt_rules)
        ov_p = self.ov_tp / max(1, self.ov_pred)
        ov_r = self.ov_tp / max(1, self.gt_pairs)
        out = {
            "Understand_Acc": self.understand_correct / n,
            "Asso_Acc": self.asso_correct / n,
            "ALL_Acc": self.both_correct / n,
            "RE_P": re_p,
            "RE_R": re_r,
            "RE_F1": (2 * re_p * re_r) / max(1e-12, re_p + re_r),
            "Overall_P": ov_p,
            "Overall_R": ov_r,
            "Overall_F1": (2 * ov_p * ov_r) / max(1e-12, ov_p + ov_r),
            "lane_correctness": self.target_lane_correct / max(1, self.target_lane_present),
            "_counts": {
                "total_scenes": self.total_scenes,
                "parse_failures": self.parse_failures,
                "gt_rules": self.gt_rules, "gt_pairs": self.gt_pairs,
                "re_tp": self.re_tp, "re_pred": self.re_pred,
                "ov_tp": self.ov_tp, "ov_pred": self.ov_pred,
                "target_lane_present": self.target_lane_present,
                "target_lane_correct": self.target_lane_correct,
            },
        }
        if self.conflict_total:
            tot = sum(self.conflict_total.values())
            cor = sum(self.conflict_correct.values())
            out["conflict_resolution_acc"] = cor / max(1, tot)
            for t in sorted(self.conflict_total.keys()):
                out[f"conflict_acc_{t}"] = (
                    self.conflict_correct.get(t, 0) / max(1, self.conflict_total[t])
                )
            out["_counts"]["conflict_total"] = dict(self.conflict_total)
            out["_counts"]["conflict_correct"] = dict(self.conflict_correct)
        return out


def compute_metrics(pred_texts: list[str], gt_texts: list[str],
                    conflict_metas: Optional[list[Optional[dict]]] = None) -> dict:
    if len(pred_texts) != len(gt_texts):
        raise ValueError(f"length mismatch: {len(pred_texts)} preds vs {len(gt_texts)} gts")
    if conflict_metas is not None and len(conflict_metas) != len(pred_texts):
        raise ValueError(
            f"conflict_metas length {len(conflict_metas)} != preds {len(pred_texts)}"
        )
    m = MapDRMetric()
    if conflict_metas is None:
        conflict_metas = [None] * len(pred_texts)
    for p, g, cm in zip(pred_texts, gt_texts, conflict_metas):
        m.add(p, g, conflict_meta=cm)
    return m.compute()


__all__ = [
    "MapDRMetric", "compute_metrics",
    "parse_response", "sort_lists_in_dict",
]
