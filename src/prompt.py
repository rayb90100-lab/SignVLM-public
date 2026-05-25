"""Prompt templates for SignVLM training/inference.

路线 3 (current, discrete VLA following DriveLM ECCV 2024 Oral style):
  - rules + lane_assignment + plan{target_lane, action, reasoning}
  - all-discrete text, no trajectory token
  - use `build_prompt()` directly

路线 1 升级 (trajectory VLA following AutoVLA NeurIPS 2025 style):
  - plan additionally contains "future_trajectory": "<action_N1><action_N2>..."
  - call `build_prompt(include_trajectory=True)` once src/action_tokenizer.py exists
  - the `_TRAJECTORY_INSTRUCTION_BLOCK` constant below is the chunk to merge in
"""
from __future__ import annotations


_HEADER = """Picture 1 是一个交通标志的特写。
Picture 2 是该标志所在路口的车载前视图，图中红色折线表示车道的中心线，每条线起点附近的黄色数字（0, 1, 2, ...）是该车道的相对索引。

请完成两件事：
1. 识别 Picture 1 上标志传达的所有交通规则（rules + lane_assignment）
2. 给出基于该标志的驾驶决策（plan）

输出严格 JSON 字典，**不要输出任何额外说明**："""


_RULES_BLOCK = """  "rules": [
    {
      "attr_info": {"AllowedTransport": "...", "EffectiveDate": "...", "EffectiveTime": "...",
                    "HighSpeedLimit": "...", "LaneDirection": [...], "LaneType": "...",
                    "LowSpeedLimit": "...", "RuleIndex": "..."},
      "centerline": [车道索引]
    }
  ],"""


_PLAN_BLOCK_DISCRETE = """  "plan": {
    "target_lane": N,
    "action": "stay/change_left/change_right/decelerate",
    "reasoning": "一句话说明为什么这样规划"
  }"""


# Path 路线 1 升级用 — append `future_trajectory` field.
# When ActionTokenizer is in place, swap _PLAN_BLOCK_DISCRETE for this and
# document `<action_N>` in the trailing notes.
_PLAN_BLOCK_WITH_TRAJECTORY = """  "plan": {
    "target_lane": N,
    "action": "stay/change_left/change_right/decelerate",
    "reasoning": "一句话说明为什么这样规划",
    "future_trajectory": "<action_N1><action_N2>...（用 motion-primitive token 序列编码未来 3s 轨迹）"
  }"""


_NOTES = """attr_info 8 字段必须都出现；不适用的填字符串 "None"。LaneDirection 是字符串列表。centerline / target_lane 是 Picture 2 上的黄色数字。"""


def build_prompt(include_trajectory: bool = False) -> str:
    """Construct the user-side prompt text.

    Parameters
    ----------
    include_trajectory : bool
        False (default, 路线 3): plan field is target_lane / action / reasoning only.
        True (路线 1 升级 once ActionTokenizer exists): plan additionally contains
        future_trajectory: "<action_N1><action_N2>...".
    """
    plan_block = _PLAN_BLOCK_WITH_TRAJECTORY if include_trajectory else _PLAN_BLOCK_DISCRETE
    return f"""{_HEADER}

{{
{_RULES_BLOCK}
{plan_block}
}}

{_NOTES}"""


# Module-level default for convenience and back-compat with src/dataset.py
PROMPT_TEMPLATE = build_prompt(include_trajectory=False)


__all__ = ["build_prompt", "PROMPT_TEMPLATE"]
