"""Offline canonical re-evaluation from preds_shard*.jsonl.

Why: DPO over-optim drifts attr_info string-field tokens (extra spaces,
case, JSON key renames) while leaving semantics intact. The original
multiset metric counts every drifted character as a miss → Tab 4 row 5
collapsed to 0 despite vision-faithful semantics. This script applies a
conservative canonicalization (strip + whitespace collapse + lowercase
on string values; lowercase on dict keys) before the multiset match,
so format noise stops killing semantic correctness.

Run:
  python scripts/eval_canonical.py <eval_dir> [--rule-version v1]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metric import parse_response, sort_lists_in_dict, _multiset_intersect  # type: ignore


# ---------------- canonical rules (versioned, conservative) ----------------

_WS_RE = re.compile(r"\s+")
# v2: any digit:digit pair where the LHS is a single digit followed by ':' and
# preceded by a non-digit boundary → pad with leading zero. Matches "7:00" but
# not "17:00" or "h7:00". Safe under "info-invariant": 7:00 and 07:00 are the
# same instant.
_TIME_LEADING_ZERO_RE = re.compile(r"(?<!\d)(\d):")
# v3: collapse runs of ';;' → ';' (separator with empty span carries no info).
_MULTI_SEMI_RE = re.compile(r";+")


def _canon_str_v1(s: str) -> str:
    """v1: strip all whitespace + lowercase."""
    return _WS_RE.sub("", s).lower()


def _canon_str_v2(s: str) -> str:
    """v2 = v1 + pad leading zero in single-digit hours.

    Examples:
      '07:00-09:00' → '07:00-09:00'   (unchanged)
      '7:00-9:00'   → '07:00-09:00'   (padded)
      '17:00'       → '17:00'         (two-digit hour preserved)
      'a:b'         → 'a:b'           (not a digit; untouched)
    """
    return _TIME_LEADING_ZERO_RE.sub(r"0\1:", _WS_RE.sub("", s).lower())


def _canon_str_v3(s: str) -> str:
    """v3 = v2 + collapse multi-semicolons + strip trailing semicolons.

    Semicolons separate windows/dates ('07:00-09:00;17:00-19:00'). A trailing
    or doubled semicolon delimits an empty span which carries no information.

    Examples:
      '17:00-19:00;'         → '17:00-19:00'
      '07:00-09:00;;16:00'   → '07:00-09:00;16:00'
      '07:00-09:00;16:00-19:00' → '07:00-09:00;16:00-19:00'   (unchanged)
    """
    s = _WS_RE.sub("", s).lower()
    s = _TIME_LEADING_ZERO_RE.sub(r"0\1:", s)
    s = _MULTI_SEMI_RE.sub(";", s).strip(";")
    return s


def _dedup_preserve_order(seq: list) -> list:
    """Drop consecutive-equal and globally repeated entries; preserve first occurrence.

    Safe under "info-invariant" because list elements in attr_info encode set
    membership (a direction value cannot legitimately repeat), and dedup neither
    adds nor removes a distinct element.
    """
    out: list = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


_STR_CANON = {"v1": _canon_str_v1, "v2": _canon_str_v2, "v3": _canon_str_v3}
_DEDUP_RULES = {"v2", "v3"}
_KEY_STRIP_RULES = {"v2", "v3"}


def _canon_value(v: Any, rule: str) -> Any:
    if isinstance(v, str):
        fn = _STR_CANON.get(rule)
        if fn is None:
            raise ValueError(f"unknown canonical rule: {rule}")
        return fn(v)
    if isinstance(v, list):
        inner = [_canon_value(x, rule) for x in v]
        if rule in _DEDUP_RULES:
            inner = _dedup_preserve_order(inner)
        return inner
    if isinstance(v, dict):
        return canonicalize_attrs(v, rule)
    return v  # ints / floats / bool / None — untouched


def _canon_key(k: Any, rule: str) -> Any:
    """Canonicalize a dict key. v1: lowercase only. v2/v3: also strip internal whitespace."""
    if not isinstance(k, str):
        return k
    if rule in _KEY_STRIP_RULES:
        return _WS_RE.sub("", k).lower()
    return k.lower()


def canonicalize_attrs(attrs: dict, rule: str = "v1") -> dict:
    """Canonicalize an attr_info dict: keys + string values per rule."""
    out: dict = {}
    for k, v in attrs.items():
        out[_canon_key(k, rule)] = _canon_value(v, rule)
    return out


# ---------------- metric (mirrors src.metric.MapDRMetric, with canon hook) ----------------

def _attr_key(attr: dict) -> str:
    return json.dumps(attr, sort_keys=True, ensure_ascii=False)


def compute_canonical(records: list[dict], rule: str = "v1") -> dict:
    """records: list of {pred_text, gt_text, conflict_meta?}"""
    total_scenes = 0
    understand_correct = 0
    asso_correct = 0
    both_correct = 0
    parse_failures = 0
    re_tp = re_pred = gt_rules = 0
    ov_tp = ov_pred = gt_pairs = 0
    target_lane_correct = target_lane_present = 0
    conflict_total: dict[str, int] = {}
    conflict_correct: dict[str, int] = {}
    fail_samples: list[dict] = []  # for v2 rule-design feedback

    for rec in records:
        gt = parse_response(rec["gt_text"])
        if gt is None:
            continue  # malformed GT — skip (matches src.metric.add behavior)
        total_scenes += 1
        gt = sort_lists_in_dict(gt)
        gt_rules_l = gt.get("rules", []) or []
        gt_attr_keys = [
            _attr_key(canonicalize_attrs(r.get("attr_info", {}) or {}, rule))
            for r in gt_rules_l
        ]
        gt_pairs_l: list[tuple[str, Any]] = []
        for r in gt_rules_l:
            ak = _attr_key(canonicalize_attrs(r.get("attr_info", {}) or {}, rule))
            for c in r.get("centerline", []) or []:
                gt_pairs_l.append((ak, c))
        gt_rules += len(gt_attr_keys)
        gt_pairs += len(gt_pairs_l)

        gt_plan = gt.get("plan", {}) or {}
        gt_tl = gt_plan.get("target_lane", -1)
        if gt_tl != -1:
            target_lane_present += 1

        cm = rec.get("conflict_meta")
        ctype = cm.get("type") if cm else None
        if ctype is not None:
            conflict_total[ctype] = conflict_total.get(ctype, 0) + 1

        pred = parse_response(rec["pred_text"])
        if pred is None:
            parse_failures += 1
            continue
        pred = sort_lists_in_dict(pred)
        pred_rules_l = pred.get("rules", []) or []
        pred_attr_keys = [
            _attr_key(canonicalize_attrs(r.get("attr_info", {}) or {}, rule))
            for r in pred_rules_l
        ]
        pred_pairs_l: list[tuple[str, Any]] = []
        for r in pred_rules_l:
            ak = _attr_key(canonicalize_attrs(r.get("attr_info", {}) or {}, rule))
            for c in r.get("centerline", []) or []:
                pred_pairs_l.append((ak, c))

        re_tp += _multiset_intersect(pred_attr_keys, gt_attr_keys)
        re_pred += len(pred_attr_keys)
        ov_tp += _multiset_intersect(pred_pairs_l, gt_pairs_l)
        ov_pred += len(pred_pairs_l)

        understand_ok = sorted(pred_attr_keys) == sorted(gt_attr_keys)
        pred_cls = sorted(tuple(r.get("centerline", []) or []) for r in pred_rules_l)
        gt_cls = sorted(tuple(r.get("centerline", []) or []) for r in gt_rules_l)
        asso_ok = pred_cls == gt_cls
        if understand_ok:
            understand_correct += 1
        if asso_ok:
            asso_correct += 1
        if understand_ok and asso_ok:
            both_correct += 1

        pred_plan = pred.get("plan", {}) or {}
        pred_tl = pred_plan.get("target_lane", -1)
        if gt_tl != -1 and pred_tl == gt_tl:
            target_lane_correct += 1

        if ctype is not None and understand_ok and asso_ok:
            conflict_correct[ctype] = conflict_correct.get(ctype, 0) + 1

        # capture understand-fail samples for v2 rule-design feedback
        if (not understand_ok) and len(fail_samples) < 40:
            fail_samples.append({
                "scene_id": rec.get("scene_id", "?")[:12],
                "conflict_type": ctype,
                "gt_attrs_canon": [json.loads(k) for k in gt_attr_keys],
                "pred_attrs_canon": [json.loads(k) for k in pred_attr_keys],
                "gt_attrs_raw": [r.get("attr_info") for r in gt_rules_l],
                "pred_attrs_raw": [r.get("attr_info") for r in pred_rules_l],
            })

    n = max(1, total_scenes)
    re_p = re_tp / max(1, re_pred)
    re_r = re_tp / max(1, gt_rules)
    ov_p = ov_tp / max(1, ov_pred)
    ov_r = ov_tp / max(1, gt_pairs)
    out: dict = {
        "rule_version": rule,
        "metrics": {
            "Understand_Acc": understand_correct / n,
            "Asso_Acc": asso_correct / n,
            "ALL_Acc": both_correct / n,
            "RE_P": re_p,
            "RE_R": re_r,
            "RE_F1": (2 * re_p * re_r) / max(1e-12, re_p + re_r),
            "Overall_P": ov_p,
            "Overall_R": ov_r,
            "Overall_F1": (2 * ov_p * ov_r) / max(1e-12, ov_p + ov_r),
            "lane_correctness": target_lane_correct / max(1, target_lane_present),
            "_counts": {
                "total_scenes": total_scenes,
                "parse_failures": parse_failures,
                "gt_rules": gt_rules, "gt_pairs": gt_pairs,
                "re_tp": re_tp, "re_pred": re_pred,
                "ov_tp": ov_tp, "ov_pred": ov_pred,
                "target_lane_present": target_lane_present,
                "target_lane_correct": target_lane_correct,
            },
        },
    }
    if conflict_total:
        tot = sum(conflict_total.values())
        cor = sum(conflict_correct.values())
        out["metrics"]["conflict_resolution_acc"] = cor / max(1, tot)
        for t in sorted(conflict_total.keys()):
            out["metrics"][f"conflict_acc_{t}"] = (
                conflict_correct.get(t, 0) / max(1, conflict_total[t])
            )
        out["metrics"]["_counts"]["conflict_total"] = dict(conflict_total)
        out["metrics"]["_counts"]["conflict_correct"] = dict(conflict_correct)
    return out, fail_samples


def load_records(eval_dir: Path) -> list[dict]:
    shards = sorted(eval_dir.glob("preds_shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no preds_shard*.jsonl in {eval_dir}")
    records: list[dict] = []
    for sp in shards:
        for line in sp.open():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="dir containing preds_shard*.jsonl")
    ap.add_argument("--rule-version", default="v1")
    ap.add_argument("--no-fail-dump", action="store_true",
                    help="skip writing fail_samples_<rule>.json")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    records = load_records(eval_dir)
    print(f"loaded {len(records)} records from {eval_dir.name}")
    result, fail_samples = compute_canonical(records, rule=args.rule_version)

    out_path = eval_dir / f"metrics_canonical_{args.rule_version}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out_path.name}")

    m = result["metrics"]
    print()
    print(f"  rule           = {result['rule_version']}")
    print(f"  Understand_Acc = {m['Understand_Acc']:.4f}")
    print(f"  Asso_Acc       = {m['Asso_Acc']:.4f}")
    print(f"  ALL_Acc        = {m['ALL_Acc']:.4f}")
    print(f"  RE_F1          = {m['RE_F1']:.4f}  (P {m['RE_P']:.3f}  R {m['RE_R']:.3f})")
    print(f"  Overall_F1     = {m['Overall_F1']:.4f}  (P {m['Overall_P']:.3f}  R {m['Overall_R']:.3f})")
    print(f"  lane_corr      = {m['lane_correctness']:.4f}")
    if "conflict_resolution_acc" in m:
        print(f"  conflict_res   = {m['conflict_resolution_acc']:.4f}")
        for k in sorted(m):
            if k.startswith("conflict_acc_"):
                print(f"    {k:25s} = {m[k]:.4f}")
    print(f"  parse_failures = {m['_counts']['parse_failures']}/{m['_counts']['total_scenes']}")

    if not args.no_fail_dump and fail_samples:
        fp = eval_dir / f"fail_samples_{args.rule_version}.json"
        fp.write_text(json.dumps(fail_samples, indent=2, ensure_ascii=False))
        print(f"wrote {fp.name} ({len(fail_samples)} samples) — inspect to design next rule")


if __name__ == "__main__":
    main()
