"""LoRA SFT training for SignVLM Stage 1 (路线 3 discrete VLA).

Backbone : Qwen2.5-VL-7B-Instruct
Method   : PEFT LoRA (rank 64) + bf16 + gradient checkpointing
Data     : src/dataset.py MapDRDataset (Train split)
Loss mask: only supervise the assistant response (system + user tokens
           are masked with -100).

PyTorch Lightning / FSDP intentionally NOT used — single-machine LoRA
finetune is short enough to write the train loop by hand, which is more
interpretable than wrapping pl.Trainer.

Usage:
  python scripts/train_sft.py [--device cuda:0] [--epochs 1] [--lora-rank 64] \\
      [--batch-size 1] [--grad-accum 8] [--lr 2e-4] [--max-steps 0] \\
      [--save-every 500] [--log-every 10] [--shuffle-idx] [--map-perturb 0.0]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Liger-Kernel: fused_linear_cross_entropy avoids materializing the
# (B, T, vocab=151936) logits tensor — saves ~1-2 GB at the cross-entropy step,
# which is what hits OOM under DDP with bf16 training. Patch applied globally
# before any Qwen2.5-VL model is constructed.
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
    apply_liger_kernel_to_qwen2_5_vl(
        rope=True,
        cross_entropy=False,           # mutually exclusive with fused_linear_cross_entropy
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
    )
    _LIGER = True
except ImportError:
    _LIGER = False

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dataset import MapDRDataset  # noqa: E402

DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # overridden via --model-name (e.g. Qwen2.5-VL-3B-Instruct for the 3B twin runs)
MODEL_DIR = REPO / "ckpts" / DEFAULT_MODEL_NAME  # module-level default; main() may rebind via args.model_name
DATA_ROOT = REPO / "data" / "MapDR"

SYSTEM_PROMPT = (
    "You are a sign-aware autonomous-driving assistant. "
    "Read the traffic sign and the rendered front view, then output a JSON dict "
    "with rules / lane_assignment / plan as instructed."
)

# Qwen2.5-VL chat-template marker for the start of an assistant turn:
# tokenizer encodes "<|im_start|>assistant" as [151644, 77091]
# Loss is only computed on tokens AFTER this marker (mirrors AutoVLA L382-419).
ASSISTANT_MARKER = torch.tensor([151644, 77091])
IGNORE_INDEX = -100


def _resize_for_qwen(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale a PIL image so that w*h ≤ max_pixels, with both sides rounded
    to multiples of 28 (Qwen2.5-VL patch size). Aspect ratio preserved."""
    w, h = img.size
    if w * h <= max_pixels:
        # still round to 28 to avoid awkward processor padding
        new_w = max(28, (w // 28) * 28)
        new_h = max(28, (h // 28) * 28)
        if (new_w, new_h) != (w, h):
            return img.resize((new_w, new_h), Image.BICUBIC)
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(28, int(w * scale) // 28 * 28)
    new_h = max(28, int(h * scale) // 28 * 28)
    return img.resize((new_w, new_h), Image.BICUBIC)


class Collator:
    """Turn a list of MapDRDataset items into model-ready tensors.

    Each item has: cropped_sign / visual_prompt (PIL.Image) + prompt_text + gt_text.
    We build a 3-turn conversation (system / user with 2 images + prompt / assistant
    with gt_text), apply the chat template, then mask everything before the
    assistant marker with IGNORE_INDEX.

    `max_image_pixels` caps each image's pixel count; visual_prompt 1920×1240
    (2.38M) is downscaled to keep memory bounded. With 1.6M default each image
    becomes ~1740×1130 (≈2048 patches @ 28×28).
    """
    def __init__(self, processor, max_image_pixels: int = 28 * 28 * 2048):
        self.processor = processor
        self.max_image_pixels = max_image_pixels

    def __call__(self, items):
        texts, images_per_sample = [], []
        for it in items:
            sign = _resize_for_qwen(it["cropped_sign"], self.max_image_pixels)
            vp = _resize_for_qwen(it["visual_prompt"], self.max_image_pixels)
            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [
                     {"type": "image", "image": sign},
                     {"type": "image", "image": vp},
                     {"type": "text", "text": it["prompt_text"]},
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": it["gt_text"]}]},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images_per_sample.append([sign, vp])

        inputs = self.processor(
            text=texts,
            images=images_per_sample,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        # Find the FIRST occurrence of [151644, 77091] (start of assistant turn);
        # mask everything up to and including it. Loss is on the actual assistant
        # response tokens (and the trailing <|im_end|>) only.
        marker = ASSISTANT_MARKER
        for i in range(labels.shape[0]):
            ids = labels[i]
            j_found = -1
            for j in range(len(ids) - len(marker) + 1):
                if torch.equal(ids[j:j + len(marker)], marker):
                    j_found = j
                    break
            if j_found < 0:
                labels[i, :] = IGNORE_INDEX  # malformed sample — fully mask
            else:
                labels[i, :j_found + len(marker)] = IGNORE_INDEX
            # also mask pad tokens (won't be backproppped anyway, but keeps log clean)
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = IGNORE_INDEX
        inputs["labels"] = labels
        return inputs


def build_args():
    p = argparse.ArgumentParser()
    # `--device` is now ignored when launched via `accelerate launch` — Accelerator
    # picks the local rank from CUDA_VISIBLE_DEVICES. Kept for single-process runs.
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0,
                   help="If >0, stop after this many optimizer steps (sanity / dry-run).")
    p.add_argument("--save-every", type=int, default=500,
                   help="Save LoRA adapter every N optimizer steps (in addition to end-of-epoch).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--shuffle-idx", action="store_true",
                   help="Enable training-time shuffling of relative centerline indices (recommended).")
    p.add_argument("--map-perturb", type=float, default=0.0,
                   help="DEPRECATED back-compat alias for --perturb-mode noise --perturb-prob <v>. "
                        "Per-sample HD-Map perturbation probability (0.0 to 1.0).")
    p.add_argument("--perturb-mode", choices=["none", "noise", "conflict"], default="none",
                   help="Map perturbation regime. 'none' = clean; 'noise' = generic input+GT "
                        "passthrough perturb (Task 6.9 ablation); 'conflict' = vision-faithful "
                        "explicit vision-map conflict, GT stays visual (W2 main path).")
    p.add_argument("--perturb-prob", type=float, default=0.0,
                   help="Per-sample probability that perturbation is applied. Ignored when "
                        "--perturb-mode none. Used together with --perturb-mode.")
    p.add_argument("--conflict-type-weights", default="",
                   help="Comma-separated weights for 4 conflict types in dataset default order "
                        "[speed,direction,vehicle,time]. Used only with --perturb-mode conflict. "
                        "Empty = uniform. Example for B-experiment (direction up-weighted): "
                        "'0.15,0.55,0.15,0.15' (direction = 55%).")
    p.add_argument("--max-image-pixels", type=int, default=28 * 28 * 2048,
                   help="Cap per-image pixel count (default option b: ~1.6M, ≤2048 patches/image)")
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                   help="Optimizer choice. 'adamw8bit' uses bitsandbytes (saves "
                        "2-4 GB optim state — needed for option-b max_pixels on 4090).")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA: load base in NF4 + double quant + bf16 compute. "
                        "Adds the SFT-4bit baseline row for the DPO-4bit Tab 4 "
                        "comparison (controls for the base-quantization "
                        "confound when row 5 is DPO-4bit).")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                   help="HF / modelscope model dir name under ckpts/ (e.g. "
                        "'Qwen2.5-VL-3B-Instruct' for the 3B twin runs).")
    p.add_argument("--out-dir", default=str(REPO / "runs" / "sft"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = build_args()
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16")
    is_main = accelerator.is_main_process

    # Race-safe run_id (see train_dpo.py same fix): same-second launches → same
    # base id → race on dir contents. Append PID for uniqueness + while-exists
    # fallback for the rare same-second-same-PID case.
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
        print(f"[run] out_dir = {out_dir}  (world_size={accelerator.num_processes})")
    accelerator.wait_for_everyone()

    # 1) processor + base model -------------------------------------------------
    # Override module-level MODEL_DIR with the per-run --model-name choice so
    # 3B/7B twin runs share the same code path. Default = 7B-Instruct.
    MODEL_DIR = REPO / "ckpts" / args.model_name  # noqa: F811 (intentional rebind)
    if is_main:
        print(f"[load] processor + model from {MODEL_DIR}")
        print(f"[load] liger_kernel patch applied: {_LIGER}  (fused_linear_cross_entropy=True)")
        print(f"[load] 4bit quantization: {args.load_in_4bit}")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), use_fast=True)
    base_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_kwargs["quantization_config"] = bnb_config
        base_kwargs["device_map"] = {"": accelerator.local_process_index}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL_DIR), **base_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if is_main:
        print(f"[load] done in {time.time() - t0:.1f}s")

    # 2) wrap with LoRA --------------------------------------------------------
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if is_main:
        model.print_trainable_parameters()

    # 3) data ------------------------------------------------------------------
    # Resolve perturb config: new API (--perturb-mode/--perturb-prob) takes precedence;
    # fall back to --map-perturb (deprecated, == noise mode).
    perturb_mode = args.perturb_mode
    perturb_prob = args.perturb_prob
    if perturb_mode == "none" and args.map_perturb > 0:
        perturb_mode = "noise"
        perturb_prob = args.map_perturb
    if perturb_mode != "none" and args.map_perturb > 0 and args.perturb_mode != "none":
        raise SystemExit("--map-perturb (deprecated) and --perturb-mode are mutually exclusive; "
                         "use only --perturb-mode/--perturb-prob.")
    # Parse conflict-type-weights (must align with dataset.conflict_types default order)
    conflict_type_weights = None
    if args.conflict_type_weights.strip():
        if perturb_mode != "conflict":
            raise SystemExit("--conflict-type-weights requires --perturb-mode conflict.")
        conflict_type_weights = [float(w) for w in args.conflict_type_weights.split(",")]
        if len(conflict_type_weights) != 4:
            raise SystemExit(f"--conflict-type-weights needs 4 floats, got "
                             f"{len(conflict_type_weights)}.")
    train_ds = MapDRDataset(
        root=DATA_ROOT,
        split="Train",
        shuffle_idx=args.shuffle_idx,
        perturb_mode=perturb_mode,
        perturb_prob=perturb_prob,
        conflict_type_weights=conflict_type_weights,
        seed=args.seed,
    )
    if is_main:
        print(f"[data] train size: {len(train_ds)} (shuffle_idx={args.shuffle_idx}, "
              f"perturb_mode={perturb_mode}, perturb_prob={perturb_prob}, "
              f"conflict_type_weights={conflict_type_weights})")
    collate = Collator(processor, max_image_pixels=args.max_image_pixels)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=False,
    )

    # 4) optimizer + scheduler -------------------------------------------------
    optim_params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            optim_params, lr=args.lr, weight_decay=args.weight_decay)
        if is_main:
            print("[optim] bitsandbytes AdamW8bit (8-bit optim state)")
    else:
        optimizer = torch.optim.AdamW(
            optim_params, lr=args.lr, weight_decay=args.weight_decay)
        if is_main:
            print("[optim] torch AdamW (fp32 optim state)")

    # Hand model / optimizer / loader to Accelerate (DDP wrap, sampler split, dtype move)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # After prepare(), each process sees its shard of the data. Steps are computed in
    # global terms (each process performs the same number of optimizer steps).
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    if is_main:
        print(f"[plan] steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
              f"effective_batch={args.batch_size * args.grad_accum * accelerator.num_processes}")

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.1, 1.0 - progress)

    # 5) train loop ------------------------------------------------------------
    model.train()
    log_f = None
    if is_main:
        log_f = open(out_dir / "train_log.jsonl", "w")
    global_step = 0
    accum_loss_sum = 0.0
    accum_loss_count = 0
    t_step = time.time()

    pbar = tqdm(total=total_steps, disable=not is_main, dynamic_ncols=True,
                desc="train", smoothing=0.0)

    stop = False
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(optim_params, 1.0)
                    for g in optimizer.param_groups:
                        g["lr"] = args.lr * lr_at(global_step)
                optimizer.step()
                optimizer.zero_grad()

                # accumulate loss for logging — only step here when sync_gradients
                accum_loss_sum += float(loss.detach())
                accum_loss_count += 1

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            pbar.update(1)
            if is_main:
                avg_loss = accum_loss_sum / max(1, accum_loss_count)
                pbar.set_postfix(loss=f"{avg_loss:.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            if global_step % args.log_every == 0 or global_step == 1:
                if is_main:
                    dt = time.time() - t_step
                    msg = {
                        "step": global_step, "epoch": epoch,
                        "loss": round(accum_loss_sum / max(1, accum_loss_count), 4),
                        "lr": optimizer.param_groups[0]["lr"],
                        "step_time_s": round(dt / max(1, args.log_every), 2),
                    }
                    log_f.write(json.dumps(msg) + "\n")
                    log_f.flush()
                    t_step = time.time()
                accum_loss_sum = 0.0
                accum_loss_count = 0

            if args.save_every and global_step % args.save_every == 0:
                accelerator.wait_for_everyone()
                ckpt_dir = out_dir / f"step_{global_step}"
                if is_main:
                    accelerator.unwrap_model(model).save_pretrained(str(ckpt_dir))
                    pbar.write(f"[ckpt] step {global_step} → {ckpt_dir}")

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    # final save (main only)
    accelerator.wait_for_everyone()
    pbar.close()
    if is_main:
        final_dir = out_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        log_f.close()
        print(f"[done] global_step={global_step}, final adapter → {final_dir}")


if __name__ == "__main__":
    main()
