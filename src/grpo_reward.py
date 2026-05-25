"""GRPO reward functions for SignVLM conflict-resolution training.

Reward signal: scene-level `conflict_resolution_acc` — 1.0 iff the model's
completion matches the GT on BOTH understand (attr_info multiset) AND asso
(centerline multiset). This is exactly the main eval metric (metric.py
MapDRMetric.add CAVP path), so reward ↔ metric is identity (Sec 2.2 of
RFT_GRPO_DESIGN.md).

Variants:
  - `conflict_reward_binary`  (default): 0/1 scene-level match
  - `conflict_reward_graded`  (ablation): per-field accumulate, [0,1] continuous

Both honor `rule_version` for syntactic normalize (v3) vs raw compare. v3
collapses formatting drift ('07:00' ↔ '7:00 ', BusLane ↔ 'Bus Lane') without
permitting vocab equivalence (中英文 / 缩写 stay as errors). See progress.md
2026-05-13 Canonical metric 设计原则段.

Used by train_grpo.py main loop:
  rewards = [conflict_reward_binary(comp, gt, meta, rule_version='raw')
             for comp, gt, meta in zip(completions, gt_texts, conflict_metas)]
"""
from __future__ import annotations
from typing import Optional, Callable, Any
import json
import re

# Reuse metric.py parse/normalize. Single source of truth.
from metric import parse_response, sort_lists_in_dict, _attr_key


# ---------------------------------------------------------------------------
# Canonical v3 normalize — same rules as scripts/eval_canonical.py v3.
# Kept in-sync; if eval_canonical.py changes, mirror here.
# ---------------------------------------------------------------------------

def _v3_normalize_str(s: str) -> str:
    """v3 normalize: strip + lowercase + semicolon collapse. Idempotent."""
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    s = re.sub(r"\s*;\s*", ";", s)
    s = re.sub(r";+", ";", s)
    s = s.strip(";")
    return s


def _v3_normalize_value(v: Any) -> Any:
    """v3 normalize one attr_info value: list dedup + time leading-zero +
    string normalize. Mirror of eval_canonical.py v3 logic."""
    if isinstance(v, str):
        # Time string '7:00-9:00' → '07:00-09:00' (leading zero on hours)
        if re.fullmatch(r"\d{1,2}:\d{2}([-~]\d{1,2}:\d{2})?", v.strip()):
            parts = re.split(r"([-~])", v.strip())
            normed = []
            for p in parts:
                if re.fullmatch(r"\d{1,2}:\d{2}", p):
                    h, m = p.split(":")
                    normed.append(f"{int(h):02d}:{m}")
                else:
                    normed.append(p)
            return _v3_normalize_str("".join(normed))
        return _v3_normalize_str(v)
    if isinstance(v, list):
        # dedup preserving order, then sort by str repr for stable compare
        seen = set()
        deduped = []
        for item in v:
            normed = _v3_normalize_value(item)
            key = json.dumps(normed, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                deduped.append(normed)
        return deduped
    if isinstance(v, dict):
        return {_v3_normalize_str(k) if isinstance(k, str) else k:
                _v3_normalize_value(vv) for k, vv in v.items()}
    return v


def _canonicalize_attr_info(attr: dict, rule_version: str = "raw") -> dict:
    """Return attr_info dict canonicalized per rule_version."""
    if rule_version == "raw":
        return attr
    if rule_version == "v3":
        return _v3_normalize_value(attr)
    raise ValueError(f"unknown rule_version: {rule_version!r}")


# ---------------------------------------------------------------------------
# Scene-level match (shared by binary + graded rewards).
# ---------------------------------------------------------------------------

def _scene_match_flags(pred_text: str, gt_text: str,
                       rule_version: str = "raw") -> tuple[bool, bool, bool]:
    """Return (parse_ok, understand_ok, asso_ok) for one scene under
    rule_version. parse_ok=False short-circuits the other two to False.
    """
    gt = parse_response(gt_text)
    if gt is None:
        # GT should always parse; treat as malformed.
        return (False, False, False)
    gt = sort_lists_in_dict(gt)
    pred = parse_response(pred_text)
    if pred is None:
        return (False, False, False)
    pred = sort_lists_in_dict(pred)

    gt_rules = gt.get("rules", []) or []
    pred_rules = pred.get("rules", []) or []

    gt_attr_keys = [_attr_key(_canonicalize_attr_info(r.get("attr_info", {}), rule_version))
                    for r in gt_rules]
    pred_attr_keys = [_attr_key(_canonicalize_attr_info(r.get("attr_info", {}), rule_version))
                      for r in pred_rules]

    understand_ok = sorted(pred_attr_keys) == sorted(gt_attr_keys)

    # Centerline (asso) compare is integer list — rule_version no-op (no string).
    pred_cls = sorted(tuple(r.get("centerline", []) or []) for r in pred_rules)
    gt_cls = sorted(tuple(r.get("centerline", []) or []) for r in gt_rules)
    asso_ok = pred_cls == gt_cls

    return (True, understand_ok, asso_ok)


# ---------------------------------------------------------------------------
# Public reward functions
# ---------------------------------------------------------------------------

def conflict_reward_binary(completion_text: str, gt_text: str,
                            conflict_meta: Optional[dict],
                            rule_version: str = "raw") -> float:
    """Scene-level binary reward (default for GRPO main).

    Returns 1.0 iff:
      - conflict_meta is not None (scene IS a conflict perturb)
      - completion parses successfully
      - pred matches GT on Understand (attr_info multiset)
      - pred matches GT on Asso (centerline multiset)

    Returns 0.0 otherwise (including parse fail / non-conflict scene).

    This is exactly metric.py MapDRMetric's `conflict_correct` hit
    condition — reward and main metric are identity.
    """
    if conflict_meta is None:
        return 0.0
    parse_ok, understand_ok, asso_ok = _scene_match_flags(
        completion_text, gt_text, rule_version)
    return 1.0 if (parse_ok and understand_ok and asso_ok) else 0.0


def conflict_reward_graded(completion_text: str, gt_text: str,
                            conflict_meta: Optional[dict],
                            rule_version: str = "raw") -> float:
    """Graded reward variant (3 levels, ablation): partial credit for U / A.

    Reward = 0.5 × understand_ok + 0.5 × asso_ok ∈ {0, 0.5, 1.0}.

    Denser signal than binary (3 levels vs 2) but still scene-level. Per-
    field accumulator is a finer-grained variant — defer until binary's
    sparse-reward risk (Risk-1 in design doc) is quantified.

    Parse fail / non-conflict scene → 0.0.
    """
    if conflict_meta is None:
        return 0.0
    parse_ok, understand_ok, asso_ok = _scene_match_flags(
        completion_text, gt_text, rule_version)
    if not parse_ok:
        return 0.0
    return 0.5 * float(understand_ok) + 0.5 * float(asso_ok)


def conflict_reward_per_field_graded(completion_text: str, gt_text: str,
                                       conflict_meta: Optional[dict],
                                       rule_version: str = "raw") -> float:
    """Per-field graded reward (Variant C, ~11 levels): densest signal.

    Reward = 0.5 × per_field_attr_match + 0.5 × asso_binary

    Where:
      - per_field_attr_match = (across rules, avg fraction of attr_info
        fields where pred matches GT). Aligned with Understand_Acc but
        with fractional credit instead of binary all-or-nothing.
      - asso_binary = 1.0 iff centerline multiset matches exactly (asso
        is a set-of-int compare; partial credit on centerlines would
        violate the geometric binding signal we want to keep crisp).

    Designed to address Risk-1 (sparse signal → zero-std groups dominate
    GRPO). Goes from 2 levels (binary) → 3 (graded) → 11+ (per-field),
    so different rollouts producing minor formatting drift on long
    string fields (e.g. EffectiveTime '07:00-09:00 ' vs '07:00-09:00')
    can produce different rewards under raw, breaking ties.

    If parse fail / non-conflict / rule count mismatch → 0.0 (penalize
    structural errors; we still need schema validity as a prerequisite).
    """
    if conflict_meta is None:
        return 0.0
    gt = parse_response(gt_text)
    if gt is None:
        return 0.0
    pred = parse_response(completion_text)
    if pred is None:
        return 0.0
    gt = sort_lists_in_dict(gt)
    pred = sort_lists_in_dict(pred)

    gt_rules = gt.get("rules", []) or []
    pred_rules = pred.get("rules", []) or []
    if not gt_rules or len(pred_rules) != len(gt_rules):
        # Rule count mismatch — model produced wrong number of rules.
        # Keep this as a structural penalty (0) rather than partial.
        return 0.0

    # Pair gt_rules ↔ pred_rules by sorted order (sort_lists_in_dict already
    # sorted lists internally; rules within scene are unordered but stable
    # after sort_lists_in_dict). For exact field compare across paired rules.
    # If multi-rule scenes need smarter matching (multi-set per-key), upgrade
    # to Hungarian; for v0 use index alignment.
    per_rule_scores = []
    for gr, pr in zip(gt_rules, pred_rules):
        gt_attr = _canonicalize_attr_info(gr.get("attr_info", {}) or {}, rule_version)
        pred_attr = _canonicalize_attr_info(pr.get("attr_info", {}) or {}, rule_version)
        fields = set(gt_attr.keys())
        if not fields:
            per_rule_scores.append(0.0)
            continue
        hits = sum(1 for f in fields if pred_attr.get(f) == gt_attr.get(f))
        per_rule_scores.append(hits / len(fields))
    per_field_attr = sum(per_rule_scores) / len(per_rule_scores)

    # Asso: centerline multiset compare (binary)
    pred_cls = sorted(tuple(r.get("centerline", []) or []) for r in pred_rules)
    gt_cls = sorted(tuple(r.get("centerline", []) or []) for r in gt_rules)
    asso_match = 1.0 if pred_cls == gt_cls else 0.0

    return 0.5 * per_field_attr + 0.5 * asso_match


# ---------------------------------------------------------------------------
# Group-relative advantage (group-level std-normalize, GRPO standard).
# ---------------------------------------------------------------------------

def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group-relative advantage: a_i = (r_i - mean) / (std + eps).

    Standard GRPO normalize (DeepSeekMath / DeepSeek-R1). When all rewards
    are equal (std=0), returns zeros — that group contributes no gradient.
    Track frequency of zero-std groups as Risk-1 monitor.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    return [(r - mean) / (std + eps) for r in rewards]


# Reward function factory: returns a callable matching the train loop signature
# `(completions, gt_texts, conflict_metas) -> list[float]`.
def make_reward_fn(variant: str = "binary", rule_version: str = "raw") -> Callable:
    """Build a reward function for the train loop.

    variant: "binary" (default) | "graded"
    rule_version: "raw" (default) | "v3"
    """
    if variant == "binary":
        per_scene = conflict_reward_binary
    elif variant == "graded":
        per_scene = conflict_reward_graded
    elif variant == "per_field_graded":
        per_scene = conflict_reward_per_field_graded
    else:
        raise ValueError(f"unknown reward variant: {variant!r}")

    def reward_fn(completions: list[str], gt_texts: list[str],
                   conflict_metas: list[Optional[dict]]) -> list[float]:
        if len(completions) != len(gt_texts) or len(completions) != len(conflict_metas):
            raise ValueError(
                f"length mismatch: completions={len(completions)} "
                f"gts={len(gt_texts)} conflict_metas={len(conflict_metas)}"
            )
        return [per_scene(c, g, m, rule_version)
                for c, g, m in zip(completions, gt_texts, conflict_metas)]

    return reward_fn


__all__ = [
    "conflict_reward_binary",
    "conflict_reward_graded",
    "conflict_reward_per_field_graded",
    "compute_group_advantages",
    "make_reward_fn",
    "_scene_match_flags",  # exported for unit tests
]
