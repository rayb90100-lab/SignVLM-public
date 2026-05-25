"""Token-mask DPO helpers.

Three helpers (see docs/RFT_TOKEN_MASK_DPO_DESIGN.md):

1. `locate_field_value_spans(text, field_names)` — given a str(dict) response
   like "{'rules': [{'attr_info': {'LaneDirection': ['TurnLeft'], ...}}, ...]}"
   and a set of field names {"LaneDirection"}, return the char (start, end)
   spans of each named field's VALUE. Multi-rule scenes yield N spans per
   field; nested dicts walked recursively.

2. `find_subseq(haystack, needle)` — find the first index i s.t.
   haystack[i:i+len(needle)] == needle. Used to locate the raw response
   token subsequence inside the full chat-templated input_ids, bypassing
   image-token complications in multimodal processors.

3. `build_conflict_token_mask(input_ids, raw_response, field_names, tokenizer)`
   — composes (1) and (2): tokenizes raw_response with offset mapping,
   finds its position in input_ids, then marks each input_ids token that
   falls inside any field-value char span. Returns a torch tensor of shape
   (T,) with 1 where the token is in a conflict span, 0 elsewhere.

Design notes:
- str(dict) responses are single-line in MapDR pipeline (no embedded '\\n'),
  so AST col_offset == char offset within the string.
- field_names typically has 1 entry (the rejected_overrides field per pair),
  but supports multiple to handle multi-field control pairs in the future.
- Returns None when subsequence match fails — caller should fall back to
  vanilla full-response logp (no token mask) for that sample.
"""
from __future__ import annotations
import ast
from typing import Optional

import torch


def locate_field_value_spans(text: str, field_names: list[str] | set[str]) -> list[tuple[int, int]]:
    """Find char (start, end) spans of each named field's value in a str(dict).

    Uses AST parse; safe for nested structures, list values, None, string
    values, and integer values. Returns empty list on syntax error.

    Each field name occurrence yields one span (in source order). Multi-rule
    scenes commonly have one occurrence per rule.
    """
    if not field_names:
        return []
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return []
    fn_set = set(field_names)
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k_node, v_node in zip(node.keys, node.values):
            if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str) \
                    and k_node.value in fn_set:
                start = k_node.col_offset  # fallback if value has no col_offset
                if hasattr(v_node, "col_offset"):
                    start = v_node.col_offset
                end = getattr(v_node, "end_col_offset", None)
                if end is None:
                    continue
                spans.append((start, end))
    spans.sort()
    return spans


def find_subseq(haystack: list[int], needle: list[int]) -> int:
    """Return first index i with haystack[i:i+len(needle)] == needle, or -1.

    Naive O(N·M) scan — fine for our N ≈ 2-3k input_ids and M ≈ 100-300
    response tokens. Avoid KMP unless profiling complains.
    """
    if not needle or len(needle) > len(haystack):
        return -1
    n0 = needle[0]
    last = len(haystack) - len(needle)
    for i in range(last + 1):
        if haystack[i] != n0:
            continue
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def build_conflict_token_mask(
    input_ids: torch.Tensor | list[int],
    raw_response: str,
    field_names: list[str] | set[str],
    tokenizer,
) -> Optional[torch.Tensor]:
    """Build a 0/1 mask over input_ids marking tokens inside conflict-field values.

    Parameters
    ----------
    input_ids : (T,) tensor or 1-D list of token ids — the FULL chat-templated
        tokenized sequence (including image tokens and chat markers).
    raw_response : the str(dict) response string (assistant content only,
        before chat template wrapping).
    field_names : conflict field names whose VALUE tokens we want to keep
        (e.g. ['LaneDirection'] for a direction-conflict pair).
    tokenizer : the text tokenizer (processor.tokenizer for multimodal).

    Returns
    -------
    Tensor of shape (T,) with dtype int64, 1 at tokens inside any conflict-
    field value span, 0 elsewhere. Returns None if the response token
    subsequence can't be located in input_ids (caller should fall back).
    """
    # 1. Tokenize raw response WITH offset mapping.
    enc = tokenizer(raw_response, add_special_tokens=False, return_offsets_mapping=True)
    sub_ids: list[int] = enc["input_ids"]
    sub_offsets: list[tuple[int, int]] = enc["offset_mapping"]
    if not sub_ids:
        return None

    # 2. Locate the subsequence in input_ids.
    if isinstance(input_ids, torch.Tensor):
        haystack = input_ids.tolist()
    else:
        haystack = list(input_ids)
    sub_start = find_subseq(haystack, sub_ids)
    if sub_start < 0:
        return None

    # 3. Locate field value char spans within raw_response.
    char_spans = locate_field_value_spans(raw_response, field_names)
    if not char_spans:
        return None  # no conflict field found → caller falls back

    # 4. For each token i in sub_ids, mark if its char range overlaps any span.
    T = len(haystack)
    mask = torch.zeros(T, dtype=torch.long)
    for i, (ts, te) in enumerate(sub_offsets):
        if ts == te:  # zero-length token (e.g. some BPE prefix) — skip
            continue
        for (cs, ce) in char_spans:
            if te > cs and ts < ce:  # any overlap
                mask[sub_start + i] = 1
                break
    return mask


__all__ = [
    "locate_field_value_spans",
    "find_subseq",
    "build_conflict_token_mask",
]
