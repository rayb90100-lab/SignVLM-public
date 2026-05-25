"""Vision-map conflict perturbation for the metadata panel.

Core contract (CAVP — Conflict-Aware Visual Prompting):
  * cropped_sign image is NEVER touched (visual truth)
  * label.json is NEVER touched (GT mirrors visual truth)
  * Only the panel dict passed into `render_visual_prompt(metadata_panel=...)`
    is perturbed — the map's *claim* is what disagrees with the sign

Four conflict classes, one selected uniformly per sample at training time:
  - speed     : Map says wrong speed limit
  - direction : Map says wrong lane direction
  - vehicle   : Map says wrong allowed vehicle type
  - time      : Map says wrong effective time range

Each `apply_*_conflict` takes a clean panel dict (output of
`projection.panel_from_attr_info`) and returns
  (perturbed_panel: dict, conflict_meta: dict).
"""
from __future__ import annotations
import random
from typing import Tuple

# Used by 'direction' conflict — MapDR LaneDirection vocabulary
# (verified by scanning 2000 label.json files: GoStraight, TurnLeft,
# TurnRight, TurnAround, Forbidden are the real values; 'UTurn' does
# not appear in MapDR — it's a synonym of TurnAround in our panel
# rendering, so we keep just TurnAround here to avoid no-op conflicts).
_DIR_POOL = ["TurnLeft", "TurnRight", "GoStraight", "TurnAround", "Forbidden"]

# Used by 'vehicle' conflict — MapDR AllowedTransport vocabulary + 'any'.
# 'any' is the panel-side rendering of clean default (no special restriction).
_VEH_POOL = ["any", "Bus", "Non-Motor", "Truck", "Vehicle"]

# Used by 'time' conflict — common time-range textual values.
# Clean default is "24h"; perturbed picks something restrictive.
_TIME_POOL = ["24h", "07:00-09:00", "17:00-19:00", "Mon-Fri", "Weekend",
              "07:00-22:00"]

# Used by 'speed' conflict — fallback speed values when original is "N/A".
_SPEED_FALLBACK = [40, 60, 80, 100, 120]

# Delta applied to integer speeds when shifting them.
_SPEED_DELTA = 20


ConflictType = str  # 'speed' | 'direction' | 'vehicle' | 'time'


def apply_speed_conflict(panel: dict, rng: random.Random) -> Tuple[dict, dict]:
    """Perturb panel['Limit'].

    - 'N/A' → pick a fake speed from `_SPEED_FALLBACK` (map adds a constraint
      the sign never mentioned).
    - 'X'   → shift by ±_SPEED_DELTA (e.g. 60 → 40 or 80).
    - 'L-H' → shift both ends by +_SPEED_DELTA (e.g. 90-120 → 110-140).

    Returns
    -------
    perturbed_panel : dict
    conflict_meta   : {'type': 'speed', 'field': 'Limit',
                       'original': str, 'perturbed': str}
    """
    p = dict(panel)
    orig = p.get("Limit", "N/A")
    if orig == "N/A":
        new = str(rng.choice(_SPEED_FALLBACK))
    elif "-" in orig:
        try:
            low, high = (int(x) for x in orig.split("-"))
            new = f"{low + _SPEED_DELTA}-{high + _SPEED_DELTA}"
        except ValueError:
            new = str(rng.choice(_SPEED_FALLBACK))
    else:
        try:
            v = int(orig)
            shift = rng.choice([+_SPEED_DELTA, -_SPEED_DELTA])
            new_int = max(20, v + shift)
            if new_int == v:
                new_int = v + _SPEED_DELTA
            new = str(new_int)
        except ValueError:
            new = str(rng.choice(_SPEED_FALLBACK))
    p["Limit"] = new
    return p, {"type": "speed", "field": "Limit",
               "original": orig, "perturbed": new}


def apply_direction_conflict(panel: dict, rng: random.Random) -> Tuple[dict, dict]:
    """Perturb panel['Dir'] to a direction NOT in the original set.

    - 'N/A'             → pick any single direction
    - 'TurnLeft'        → pick from {TurnRight, GoStraight, UTurn}
    - 'TurnLeft,GoStraight' → pick from {TurnRight, UTurn}
    """
    p = dict(panel)
    orig = p.get("Dir", "N/A")
    if orig == "N/A":
        new = rng.choice(_DIR_POOL)
    else:
        orig_set = {x.strip() for x in orig.split(",") if x.strip()}
        candidates = [d for d in _DIR_POOL if d not in orig_set]
        new = rng.choice(candidates) if candidates else "UTurn"
    p["Dir"] = new
    return p, {"type": "direction", "field": "Dir",
               "original": orig, "perturbed": new}


def apply_vehicle_conflict(panel: dict, rng: random.Random) -> Tuple[dict, dict]:
    """Perturb panel['Veh'] to a different vehicle restriction.

    - 'any'           → pick a specific restriction (Bus / Non-Motor / Truck)
    - 'Bus'           → pick from {any, Non-Motor, Truck}
    - anything else   → pick something else from the pool
    """
    p = dict(panel)
    orig = p.get("Veh", "any")
    if orig == "any":
        candidates = ["Bus", "Non-Motor", "Truck"]
    else:
        orig_set = {x.strip() for x in orig.split(",") if x.strip()}
        candidates = [v for v in _VEH_POOL if v not in orig_set]
    new = rng.choice(candidates) if candidates else "Truck"
    p["Veh"] = new
    return p, {"type": "vehicle", "field": "Veh",
               "original": orig, "perturbed": new}


def apply_time_conflict(panel: dict, rng: random.Random) -> Tuple[dict, dict]:
    """Perturb panel['Time'] to a different effective range.

    Almost always shifts clean '24h' to a restrictive window. If original
    is already restricted, picks a different restriction or flips to '24h'.
    """
    p = dict(panel)
    orig = p.get("Time", "24h")
    candidates = [t for t in _TIME_POOL if t != orig]
    new = rng.choice(candidates) if candidates else "Weekend"
    p["Time"] = new
    return p, {"type": "time", "field": "Time",
               "original": orig, "perturbed": new}


_APPLY_FNS = {
    "speed": apply_speed_conflict,
    "direction": apply_direction_conflict,
    "vehicle": apply_vehicle_conflict,
    "time": apply_time_conflict,
}


def panel_to_rejected_fields(conflict_meta: dict) -> dict:
    """Reverse of `projection.panel_from_attr_info` for ONE conflict type.

    Given a `conflict_meta` (output of `apply_conflict_to_panel`), returns the
    `attr_info` overrides needed to construct a "panel-trust" rejected answer
    from a chosen (vision-truth) GT. Caller applies these overrides to every
    rule in the scene.

    Used by DPO pair construction: rejected = chosen with these fields
    overwritten → represents the answer a panel-trusting model would emit.

    Returns
    -------
    dict mapping attr_info field name → new value (str for single-value
    fields; list[str] for LaneDirection). Empty dict if conflict_type is
    unknown (defensive — caller should never see this).

    Notes
    -----
    The mapping is intentionally coarse: every rule's field is set to the
    same panel-derived value. This is the worst-case "fully trust panel"
    behavior — exactly what we want as the negative example.
    """
    ct = conflict_meta.get("type")
    new = conflict_meta.get("perturbed", "")

    if ct == "direction":
        # panel.Dir is comma-separated; LaneDirection in attr_info is list[str].
        if new in ("", "N/A"):
            return {"LaneDirection": ["None"]}
        parts = [x.strip() for x in new.split(",") if x.strip()]
        return {"LaneDirection": parts if parts else ["None"]}

    if ct == "speed":
        # panel.Limit reverse:
        #   "N/A" → both None
        #   "L-H" → Low=L, High=H
        #   "X"   → High=X, Low=None  (matches panel_from_attr_info line 314-318)
        if new in ("", "N/A"):
            return {"HighSpeedLimit": "None", "LowSpeedLimit": "None"}
        if "-" in new:
            try:
                low, high = [s.strip() for s in new.split("-", 1)]
                return {"HighSpeedLimit": high, "LowSpeedLimit": low}
            except ValueError:
                return {"HighSpeedLimit": new, "LowSpeedLimit": "None"}
        return {"HighSpeedLimit": new, "LowSpeedLimit": "None"}

    if ct == "vehicle":
        # panel.Veh reverse:
        #   "any"          → AllowedTransport="Vehicle"  (panel_from_attr_info L323)
        #   "X" or "X,Y"   → AllowedTransport=first value (every rule gets same)
        if new == "any":
            return {"AllowedTransport": "Vehicle"}
        first = new.split(",")[0].strip() if new else "Vehicle"
        return {"AllowedTransport": first or "Vehicle"}

    if ct == "time":
        # panel.Time reverse:
        #   "24h"          → EffectiveTime="None"  (panel_from_attr_info L328-330)
        #   else           → EffectiveTime=panel value, with ',' → ';' (MapDR uses ;)
        if new == "24h":
            return {"EffectiveTime": "None"}
        return {"EffectiveTime": new.replace(",", ";") if new else "None"}

    return {}


def apply_conflict_to_panel(
    panel_clean: dict,
    conflict_type: ConflictType | None = None,
    rng: random.Random | None = None,
    types_pool: list[str] | None = None,
    type_weights: list[float] | None = None,
) -> Tuple[dict, dict]:
    """Top-level entry point.

    Parameters
    ----------
    panel_clean : output of `projection.panel_from_attr_info`
    conflict_type : if None, randomly choose from `types_pool`
    rng : pass `dataset.rng` for reproducibility; falls back to `random` module
    types_pool : restrict random choice. Default = all 4 classes.
    type_weights : per-type sampling weights aligned with `types_pool` (or
        the default 4-class order when types_pool is None). None → uniform.
        Used by B-experiment (direction-up-weighted training).

    Returns
    -------
    panel_perturbed : dict
    conflict_meta : {'type': str, 'field': str, 'original': str, 'perturbed': str}
    """
    if rng is None:
        rng = random.Random()
    if conflict_type is None:
        pool = types_pool or list(_APPLY_FNS.keys())
        if type_weights is not None:
            if len(type_weights) != len(pool):
                raise ValueError(
                    f"type_weights length {len(type_weights)} != pool length {len(pool)}")
            conflict_type = rng.choices(pool, weights=type_weights, k=1)[0]
        else:
            conflict_type = rng.choice(pool)
    if conflict_type not in _APPLY_FNS:
        raise ValueError(
            f"unknown conflict_type: {conflict_type!r} "
            f"(valid: {list(_APPLY_FNS.keys())})"
        )
    return _APPLY_FNS[conflict_type](panel_clean, rng)


def apply_multi_field_conflict_to_panel(
    panel_clean: dict,
    n_fields: int,
    rng: random.Random | None = None,
    types_pool: list[str] | None = None,
) -> Tuple[dict, dict]:
    """Apply `n_fields` distinct conflict types in sequence to one panel.

    Used for n_fields∈{2,3,4} ablation (paper Sec 4 robustness). Eval-only —
    trained models never saw multi-field perturb during training.

    Returns
    -------
    panel_perturbed : dict (all n_fields fields perturbed)
    conflict_meta   : {
        'type': 'multi',
        'n_fields': int,
        'fields': list[str],          # ordered list of perturbed type names
        'metas': list[dict],          # per-field conflict_meta from each apply_*
    }
    """
    if rng is None:
        rng = random.Random()
    pool = types_pool or list(_APPLY_FNS.keys())
    if n_fields < 1 or n_fields > len(pool):
        raise ValueError(f"n_fields={n_fields} out of range [1, {len(pool)}]")
    chosen = rng.sample(pool, n_fields)  # distinct types, no repeats
    panel = panel_clean
    metas = []
    for t in chosen:
        panel, m = _APPLY_FNS[t](panel, rng)
        metas.append(m)
    return panel, {
        "type": "multi",
        "n_fields": n_fields,
        "fields": chosen,
        "metas": metas,
    }
