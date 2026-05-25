"""GRPO trainer for SignVLM (custom, NOT trl).

Why custom: trl 0.23.1 GRPOTrainer hardcodes num_images=1 in
prepare_multimodal_messages (grpo_trainer.py:1083), incompatible with our
(cropped_sign, visual_prompt) dual-image setup inherited from RuleVLM. Fork
or custom — we go custom to share the v1 plan A skeleton (peft adapter swap
for SFT-init KL anchor + paged_adamw8bit + Liger + gradient_checkpointing
use_reentrant=True).

Algorithm (per step, Phase 0 starting config):
  1. Sample 1 prompt batch (B=1 for v0)
  2. Generate G rollouts per prompt (do_sample=True, T=1.0, top_p=0.95)
  3. Score each rollout with conflict_reward_per_field_graded (11+ levels,
     verified 20% zero-std at G=8 in Phase 0 smoke)
  4. Compute group-relative std-normalized advantages
  5. For each of (B × G) (prompt, completion) pairs:
     a. Policy forward → per-completion-token logp (grad on)
     b. Ref forward (adapter swap "ref", no_grad) → ref logp
     c. surrogate = -advantage × Σ_t log π_pol(y_t)  +  λ × Σ_t ((log π_pol - log π_ref)^2 / 2)
     d. Backward(surrogate / (B × G))   ← Plan A per-sample sequential
  6. Optimizer step, log

Phase 0 starting config (5/14 03:30 verified):
  G=8, T=1.0, top_p=0.95, λ_KL=0.1, lr=5e-6, init=C0.3 SFT adapter

See docs/RFT_GRPO_DESIGN.md §2 + §4 Risk-1 mitigation log.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

# Liger-Kernel — same as DPO (no fused_linear_cross_entropy: we need raw
# logits for per-token logp computation, same reason as DPO).
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
    apply_liger_kernel_to_qwen2_5_vl(
        rope=True, cross_entropy=False, fused_linear_cross_entropy=False,
        rms_norm=True, swiglu=True,
    )
    _LIGER = True
except ImportError:
    _LIGER = False

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from grpo_dataset import MapDRGRPODataset  # noqa: E402
from grpo_reward import make_reward_fn, compute_group_advantages  # noqa: E402

# Same peft 0.19 + transformers 4.57 TP workaround as DPO trainer
import peft.utils.save_and_load as _peft_save_load  # noqa: E402
_peft_save_load._maybe_shard_state_dict_for_tp = lambda *a, **kw: None

DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
MODEL_DIR = REPO / "ckpts" / DEFAULT_MODEL_NAME

SYSTEM_PROMPT = (
    "You are a sign-aware autonomous-driving assistant. "
    "Read the traffic sign and the rendered front view, then output a JSON dict "
    "with rules / lane_assignment / plan as instructed."
)

IGNORE_INDEX = -100
_LOGP_CHUNK_TOKENS = 64  # chunked cross-entropy to avoid 1.5GB transient alloc


def _resize_for_qwen(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        new_w = max(28, (w // 28) * 28)
        new_h = max(28, (h // 28) * 28)
        if (new_w, new_h) != (w, h):
            return img.resize((new_w, new_h), Image.BICUBIC)
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(28, int(w * scale) // 28 * 28)
    new_h = max(28, int(h * scale) // 28 * 28)
    return img.resize((new_w, new_h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Dataset collator — returns raw images + chat-format prompt, no tokenization
# yet. Rollout / per-completion-logp build their own tensors from these.
# ---------------------------------------------------------------------------

class GRPOCollator:
    """B=1 minimal collator. Returns dict with raw images / prompt text /
    GT / conflict_meta for downstream rollout + reward. Not tokenizing
    here keeps rollout flexible (G generations need same prompt batched G
    times with different sample seeds).
    """
    def __init__(self, max_image_pixels: int = 28 * 28 * 1024):
        self.max_image_pixels = max_image_pixels

    def __call__(self, items: list[dict]) -> dict:
        # B=1 enforced for v0; if B>1, just collect lists.
        out = {
            "cropped_sign": [],
            "visual_prompt": [],
            "prompt_text": [],
            "gt_text": [],
            "conflict_meta": [],
            "scene_id": [],
        }
        for it in items:
            out["cropped_sign"].append(
                _resize_for_qwen(it["cropped_sign"], self.max_image_pixels))
            out["visual_prompt"].append(
                _resize_for_qwen(it["visual_prompt"], self.max_image_pixels))
            out["prompt_text"].append(it["prompt_text"])
            out["gt_text"].append(it["gt_text"])
            out["conflict_meta"].append(it["conflict_meta"])
            out["scene_id"].append(it["scene_id"])
        return out


def _build_prompt_messages(sign, vp, prompt_text):
    return [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [
             {"type": "image", "image": sign},
             {"type": "image", "image": vp},
             {"type": "text", "text": prompt_text},
         ]},
    ]


# ---------------------------------------------------------------------------
# Rollout: generate G completions per prompt with sampling.
# Returns G completion strings + their token-id sequences (right-padded).
# ---------------------------------------------------------------------------

def _generate_rollouts(model, processor, sign, vp, prompt_text,
                        num_generations: int, temperature: float, top_p: float,
                        max_new_tokens: int, device):
    """Returns: completions (list[str], len=G), gen_ids (LongTensor, (G, T_seq))
    where T_seq = prompt_len + (≤ max_new_tokens) for each G rollout."""
    messages = _build_prompt_messages(sign, vp, prompt_text)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text] * num_generations,
        images=[[sign, vp]] * num_generations,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = gen_ids[:, prompt_len:]
    completions = processor.tokenizer.batch_decode(
        completion_ids, skip_special_tokens=True)
    return completions, gen_ids, inputs, prompt_len


# ---------------------------------------------------------------------------
# Per-completion logp: forward through (prompt + completion) and sum logp
# over completion tokens only. Returns per-token logp (T_completion,) for
# downstream KL compute, plus the scalar Σ logp for the policy surrogate.
# ---------------------------------------------------------------------------

def _per_completion_logp(model, full_input_ids, attention_mask,
                          pixel_values, image_grid_thw, prompt_len: int,
                          completion_len: int):
    """Forward (prompt + completion) once, return (T_completion,) tensor of
    per-token log π(y_t | y_<t).

    full_input_ids: (1, T_seq) — prompt tokens followed by completion tokens
    prompt_len: int — how many prompt tokens at the front (no logp credit)
    completion_len: int — how many real (non-pad) completion tokens

    Returns: per_token_logp (1, completion_len) float32
    """
    out = model(input_ids=full_input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    logits = out.logits  # (1, T_seq, V) bf16

    # We want per-token logp of completion tokens y_t = full_input_ids[:, prompt_len:prompt_len+completion_len]
    # Standard shift: predict y_t from logits[:, t-1, :]. So slice:
    #   logits at positions [prompt_len - 1, prompt_len, ..., prompt_len + completion_len - 2]
    #   labels at positions [prompt_len,     prompt_len + 1, ..., prompt_len + completion_len - 1]
    pred_logits = logits[:, prompt_len - 1: prompt_len + completion_len - 1, :]  # (1, T_c, V)
    target_ids = full_input_ids[:, prompt_len: prompt_len + completion_len]      # (1, T_c)

    B, Tc, V = pred_logits.shape
    flat_logits = pred_logits.reshape(-1, V)
    flat_targets = target_ids.reshape(-1)
    N = flat_logits.shape[0]
    per_tok = torch.empty(N, dtype=torch.float32, device=logits.device)
    for s in range(0, N, _LOGP_CHUNK_TOKENS):
        e = min(s + _LOGP_CHUNK_TOKENS, N)
        nll_chunk = F.cross_entropy(flat_logits[s:e], flat_targets[s:e], reduction="none")
        per_tok[s:e] = -nll_chunk
    return per_tok.view(B, Tc)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-adapter", default="",
                   help="Path to SFT adapter for warm-start (e.g. C0.3). Empty = fresh LoRA.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1,
                   help="v0 enforces B=1 (per-sample sequential backward)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0,
                   help="0 = full epoch loop; >0 = cap step count (dry-run / smoke)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=1,
                   help="Default 1 (every step) since GRPO step cost ~80s vs SFT ~5s")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = main-thread loading (images already PIL, no big preprocessing)")
    p.add_argument("--shuffle-idx", action="store_true")
    p.add_argument("--conflict-type-weights", default="",
                   help="Comma-separated weights for [speed,direction,vehicle,time]. "
                        "Empty = uniform. Use '0.15,0.55,0.15,0.15' for B variant "
                        "(dir over-sample, CAVP framework transfer to GRPO).")

    # GRPO-specific
    p.add_argument("--group-size", "-G", type=int, default=8,
                   help="Rollouts per prompt. Phase 0 smoke verified G=8.")
    p.add_argument("--temperature", "-T", type=float, default=1.0,
                   help="Phase 0 smoke verified T=1.0 (vs 0.7 had higher zero-std).")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Trimmed from 1024 to cap step time + KV cache budget.")
    p.add_argument("--reward-variant", default="per_field_graded",
                   choices=["binary", "graded", "per_field_graded"],
                   help="Phase 0 verified per_field_graded (11+ levels, 20pct zero-std).")
    p.add_argument("--reward-rule-version", default="raw", choices=["raw", "v3"])
    p.add_argument("--kl-weight", "--lambda", type=float, default=0.1,
                   help="λ for KL anchor term; v1 DPO Phase 0 verified 0.1 keeps schema.")
    p.add_argument("--beta-grpo", type=float, default=1.0,
                   help="Scalar multiplier on policy surrogate. 1.0 = standard GRPO.")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient norm clipping threshold. DeepSeek-R1 standard 1.0. "
                        "Set to 0 or float('inf') to disable (only measure).")

    p.add_argument("--max-image-pixels", type=int, default=28 * 28 * 1024)
    p.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                   default="paged_adamw8bit",
                   help="paged_adamw8bit default — GRPO step is rollout-heavy, "
                        "keep optim memory minimal to leave room for KV cache.")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--out-dir", default=str(REPO / "runs" / "grpo"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = build_args()
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16")
    is_main = accelerator.is_main_process

    if accelerator.num_processes > 1:
        raise SystemExit("v0 train_grpo.py supports single-GPU only (plan A path)")

    torch.cuda.set_device(accelerator.local_process_index)
    device = accelerator.device

    # Rebind MODEL_DIR per args.model_name (3B/7B switch)
    global MODEL_DIR
    MODEL_DIR = REPO / "ckpts" / args.model_name

    base_id = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{base_id}_pid{os.getpid()}"
    out_dir = Path(args.out_dir) / run_id
    if is_main:
        suffix = 0
        while out_dir.exists():
            suffix += 1
            out_dir = Path(args.out_dir) / f"{run_id}_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
        print(f"[run] out_dir = {out_dir}")

    # --- load processor + model ---
    print(f"[load] processor + base from {MODEL_DIR}")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), use_fast=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    base.config.use_cache = False  # required by gradient_checkpointing
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True})

    # --- LoRA: peft adapter swap (same as DPO) ---
    if args.init_adapter:
        print(f"[lora] resume policy adapter from {args.init_adapter}")
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        print(f"[lora] load frozen ref adapter (SFT-init snapshot, KL anchor)")
        model.load_adapter(args.init_adapter, adapter_name="ref", is_trainable=False)
        # peft.load_adapter switches active to the new one; restore default explicitly
        model.set_adapter("default")
    else:
        print(f"[lora] fresh LoRA (no init adapter) — ref will be base (disable_adapter)")
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

    if _LIGER and is_main:
        print(f"[liger] enabled (rope/rms_norm/swiglu; cross_entropy=False for raw logits)")

    model.print_trainable_parameters() if is_main else None

    # --- optimizer ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "paged_adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr,
                                              weight_decay=args.weight_decay)
    elif args.optim == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr,
                                         weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                       weight_decay=args.weight_decay)

    # --- dataset / dataloader ---
    conflict_type_weights = None
    if args.conflict_type_weights.strip():
        conflict_type_weights = [float(w) for w in args.conflict_type_weights.split(",")]
        if len(conflict_type_weights) != 4:
            raise SystemExit(f"--conflict-type-weights needs 4 floats, got {len(conflict_type_weights)}.")
    ds = MapDRGRPODataset(
        root=REPO / "data" / "MapDR", split="Train",
        shuffle_idx=args.shuffle_idx, seed=args.seed,
        conflict_type_weights=conflict_type_weights,
    )
    if is_main:
        print(f"[data] train: conflict_type_weights={conflict_type_weights}  "
              f"(None=uniform; B-variant=[0.15,0.55,0.15,0.15] dir over-sample)")
    collator = GRPOCollator(max_image_pixels=args.max_image_pixels)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # LR warmup (linear)
    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr

    reward_fn = make_reward_fn(variant=args.reward_variant,
                                rule_version=args.reward_rule_version)

    # --- train loop ---
    total_steps = args.max_steps if args.max_steps > 0 else len(dataloader) * args.epochs
    print(f"[train] starting, total_steps = {total_steps}, G = {args.group_size}, "
          f"reward = {args.reward_variant} ({args.reward_rule_version}), "
          f"kl_weight = {args.kl_weight}")

    log_path = out_dir / "train_log.jsonl" if is_main else None
    pbar = tqdm(total=total_steps, disable=not is_main, desc="grpo")
    global_step = 0
    unwrapped = accelerator.unwrap_model(model)

    for epoch in range(args.epochs):
        for batch in dataloader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            t_step = time.time()

            # Adjust LR (warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)

            B = len(batch["scene_id"])
            assert B == args.batch_size == 1, "v0 enforces B=1"

            sign = batch["cropped_sign"][0]
            vp = batch["visual_prompt"][0]
            prompt_text = batch["prompt_text"][0]
            gt_text = batch["gt_text"][0]
            cmeta = batch["conflict_meta"][0]
            scene_id = batch["scene_id"][0]

            # === 1. Rollout (no_grad) ===
            model.eval()
            t_rollout = time.time()
            completions, gen_ids, gen_inputs, prompt_len = _generate_rollouts(
                model, processor, sign, vp, prompt_text,
                num_generations=args.group_size,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            rollout_time = time.time() - t_rollout

            # === 2. Reward + advantage ===
            rewards = reward_fn(
                completions,
                [gt_text] * args.group_size,
                [cmeta] * args.group_size,
            )
            advantages = compute_group_advantages(rewards)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            zero_std = std_r < 1e-6

            # === 3. Per-sample policy + ref logp, surrogate, backward ===
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            total_kl = 0.0
            total_logp_pol = 0.0
            t_backward = time.time()

            # Pre-extract per-rollout vision tensors from gen_inputs (all G
            # rollouts share the same image, so we can reuse a single
            # pixel_values / image_grid_thw for each sample).
            # gen_inputs was created with G identical (sign, vp) pairs, so
            # we can just slice rollout i's portion. But Qwen2.5-VL processor
            # interleaves pixel_values along image_count axis; easiest is to
            # re-process each rollout's (sign, vp) individually for the
            # train forward. Slightly redundant but simpler and avoids
            # subtle slicing bugs.
            for i in range(args.group_size):
                if zero_std and abs(advantages[i]) < 1e-9 and args.kl_weight == 0:
                    # No gradient contribution at all; skip forward to save time.
                    continue

                # Build training tensor: full sequence = prompt + completion[i]
                # We already have prompt_len from gen_inputs; just take the i-th
                # row of gen_ids which contains [prompt, completion_i, optional pad]
                full_seq_full = gen_ids[i:i+1]  # (1, T_seq)
                # Compute non-pad completion length
                completion_part = full_seq_full[0, prompt_len:]
                pad_id = processor.tokenizer.pad_token_id
                eos_id = processor.tokenizer.eos_token_id
                # First pad or eos terminates the real completion span.
                non_pad_mask = (completion_part != pad_id)
                if eos_id is not None:
                    # Keep eos in the sequence but stop counting after it
                    pass
                completion_len = int(non_pad_mask.sum().item())
                if completion_len < 1:
                    continue  # skip empty rollout

                # Trim to non-pad portion to avoid wasted forward
                T_eff = prompt_len + completion_len
                full_seq = full_seq_full[:, :T_eff]
                attn_mask = torch.ones_like(full_seq)

                # Re-build pixel_values for THIS rollout (same image pair)
                vis_inputs = processor(
                    text=[processor.apply_chat_template(
                        _build_prompt_messages(sign, vp, prompt_text),
                        tokenize=False, add_generation_prompt=True)],
                    images=[[sign, vp]],
                    padding=True, return_tensors="pt",
                ).to(device)
                pixel_values = vis_inputs["pixel_values"]
                image_grid_thw = vis_inputs["image_grid_thw"]

                # --- ref forward (adapter swap, no_grad) ---
                unwrapped.set_adapter("ref")
                try:
                    with torch.no_grad():
                        logp_ref = _per_completion_logp(
                            model, full_seq, attn_mask, pixel_values, image_grid_thw,
                            prompt_len=prompt_len, completion_len=completion_len,
                        )
                finally:
                    unwrapped.set_adapter("default")

                # --- policy forward (grad on) ---
                logp_pol = _per_completion_logp(
                    model, full_seq, attn_mask, pixel_values, image_grid_thw,
                    prompt_len=prompt_len, completion_len=completion_len,
                )

                # --- surrogate loss ---
                log_prob_seq = logp_pol.sum()  # scalar, Σ_t log π_pol(y_t)
                kl_full = ((logp_pol - logp_ref) ** 2 / 2.0).sum()  # scalar, Σ_t ((Δ)^2/2)
                surrogate = (
                    -args.beta_grpo * advantages[i] * log_prob_seq
                    + args.kl_weight * kl_full
                ) / args.group_size

                accelerator.backward(surrogate)
                total_loss += float(surrogate.item())
                total_kl += float(kl_full.item())
                total_logp_pol += float(log_prob_seq.item())

            backward_time = time.time() - t_backward

            # grad_norm + optional clip. RL rollout occasionally produces
            # extreme samples → grad spike → can destroy LoRA in one step.
            # DeepSeek-R1 standard max_grad_norm=1.0. Set 0/inf to disable.
            clip_at = (float("inf") if args.max_grad_norm <= 0
                        else args.max_grad_norm)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=clip_at,
            ).item())

            optimizer.step()
            global_step += 1
            step_time = time.time() - t_step

            # === 4. Log ===
            if is_main and global_step % args.log_every == 0:
                avg_kl = total_kl / args.group_size
                avg_logp = total_logp_pol / args.group_size
                rec = {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "scene_id": scene_id,
                    "conflict_type": cmeta.get("type") if cmeta else None,
                    "rewards": rewards,
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                    "zero_std": zero_std,
                    "advantages": advantages,
                    "loss": total_loss,
                    "kl_term": avg_kl,
                    "grad_norm": grad_norm,
                    "logp_pol_seq": avg_logp,
                    "rollout_time_s": rollout_time,
                    "backward_time_s": backward_time,
                    "step_time_s": step_time,
                }
                log_path.write_text(
                    (log_path.read_text() if log_path.exists() else "") +
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )
                desc = (f"step {global_step}/{total_steps} | "
                        f"r={mean_r:.3f}±{std_r:.3f} | loss={total_loss:.3f} | "
                        f"kl={avg_kl:.3f} | gn={grad_norm:.3f} | t={step_time:.1f}s")
                pbar.set_description(desc)
                pbar.update(1)

            if global_step >= total_steps:
                break

            # Periodic checkpoint
            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                ck_dir = out_dir / f"step_{global_step}"
                unwrapped.save_pretrained(str(ck_dir))
                print(f"\n[ckpt] saved → {ck_dir}")

    # Final save
    if is_main:
        final_dir = out_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        print(f"\n[done] final adapter saved → {final_dir}")
    pbar.close()


if __name__ == "__main__":
    main()
