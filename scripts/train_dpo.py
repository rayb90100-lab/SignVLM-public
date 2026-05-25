"""LoRA DPO training for SignVLM Stage 2 (vision-faithful conflict resolution).

Backbone : Qwen2.5-VL-7B-Instruct (shared between policy and reference)
Method   : PEFT LoRA + DPO (Rafailov et al., NeurIPS 2023) + bf16 + grad ckpt
Data     : src/dpo_dataset.py MapDRDPODataset — mix of conflict / control pairs
Init     : --init-adapter <path>  (default: C0.3 SFT adapter)

Why not trl.DPOTrainer (0.23.1):
  - process_row carries pixel_values / image_sizes / pixel_attention_mask
    but NOT image_grid_thw, which Qwen2.5-VL forward requires for the
    dynamic-resolution vision tower (modeling_qwen2_5_vl L1023).
  - patching 4 internal static methods is fragile across trl minor releases;
    writing the DPO loss directly (~10 lines) and reusing the train_sft.py
    accelerate / liger / lora plumbing is more robust.

DPO loss (paper Eq. 7):
  L_DPO = -log σ(β · ((log π_θ(y_w|x) - log π_ref(y_w|x))
                     - (log π_θ(y_l|x) - log π_ref(y_l|x))))

Reference policy:
  Same base + same LoRA adapter as policy, but adapter disabled at forward
  time via `peft.PeftModel.disable_adapter()`. This avoids loading a second
  copy of the 14GB base.

Per-batch flow:
  collator emits a (2B, ...) batch where the first B are chosen responses
  and the next B are rejected. One forward pass computes both per-seq logp.
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# Liger-Kernel: same plumbing as SFT. We DO NOT enable fused_linear_cross_entropy
# here because DPO needs the raw logits to compute per-token logprobs — fused
# CE materializes them only at the loss step.
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
from token_mask import build_conflict_token_mask  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
from dpo_dataset import MapDRDPODataset  # noqa: E402

# Workaround: peft 0.19 + transformers 4.57 version mismatch.
# peft's set_peft_model_state_dict calls _maybe_shard_state_dict_for_tp when
# torch.distributed is initialized; that function does `from
# transformers.integrations.tensor_parallel import EmbeddingParallel`, which
# only exists in transformers 5.x. We are not using tensor parallelism, so we
# replace the call with a no-op. Single-GPU dry-run isn't affected (the if
# branch never enters), but multi-GPU triggers the ImportError.
import peft.utils.save_and_load as _peft_save_load  # noqa: E402
_peft_save_load._maybe_shard_state_dict_for_tp = lambda *a, **kw: None

DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # overridden via --model-name (e.g. Qwen2.5-VL-3B-Instruct for the 3B twin runs)
MODEL_DIR = REPO / "ckpts" / DEFAULT_MODEL_NAME  # module-level default; main() rebinds from args.model_name
DATA_ROOT = REPO / "data" / "MapDR"

SYSTEM_PROMPT = (
    "You are a sign-aware autonomous-driving assistant. "
    "Read the traffic sign and the rendered front view, then output a JSON dict "
    "with rules / lane_assignment / plan as instructed."
)

ASSISTANT_MARKER = torch.tensor([151644, 77091])  # <|im_start|>assistant
IGNORE_INDEX = -100


def _resize_for_qwen(img: Image.Image, max_pixels: int) -> Image.Image:
    """Same as train_sft._resize_for_qwen — keep both dims as multiples of 28."""
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


class DPOCollator:
    """Build chosen/rejected batches from B pair items.

    Two output modes:

    1. **plan A (default, single_pass=False)** — returns TWO separate (B, ...)
       batches: `{"chosen": {input_ids,...,labels}, "rejected": {...}}`. Train
       loop forwards them sequentially with per-sample 2× backward, keeping
       activation peak at 1× single-sample. Required for 7B on 24GB 4090.
       Multi-GPU DDP incompatible (see progress.md 5/13 多卡探索).

    2. **single-pass (single_pass=True)** — returns ONE (2B, ...) concat batch
       (chosen[:B] + rejected[B:]). Train loop does single forward + single
       backward. Activation peak is 2× sample so OOMs 7B on 24GB long-tail,
       BUT works on multi-GPU DDP (DDP standard 1-forward-1-backward pattern).
       Use this for the 3B twin multi-card runs where the smaller base
       (~6 GB vs 7B 14 GB) leaves room for the 2× activation.

    History (2026-05-12 → 2026-05-13):
      v1 = single-pass mode, OOMed on 7B long-tail (5/12 #1-#5 战).
      v2 = plan A mode (this is the "5/13 plan A" — DPO collator chosen/rejected per-sample).
      Both kept under a flag to share code with 3B multi-card runs.
    """
    def __init__(self, processor, max_image_pixels: int = 28 * 28 * 1024,
                 single_pass: bool = False, build_conflict_mask: bool = False):
        self.processor = processor
        self.max_image_pixels = max_image_pixels
        self.single_pass = single_pass
        # If True, attach per-sample `conflict_mask` tensors marking the
        # response tokens whose chars lie inside conflict-field VALUE spans.
        # Used by token-mask DPO (v1) to restrict the reward signal.
        self.build_conflict_mask = build_conflict_mask

    def _build_messages(self, sign, vp, prompt_text, response_text):
        return [
            {"role": "system",
             "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",
             "content": [
                 {"type": "image", "image": sign},
                 {"type": "image", "image": vp},
                 {"type": "text", "text": prompt_text},
             ]},
            {"role": "assistant",
             "content": [{"type": "text", "text": response_text}]},
        ]

    def _process_one(self, texts, images):
        inputs = self.processor(
            text=texts, images=images,
            padding=True, return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        marker = ASSISTANT_MARKER
        for i in range(labels.shape[0]):
            ids = labels[i]
            j_found = -1
            for j in range(len(ids) - len(marker) + 1):
                if torch.equal(ids[j:j + len(marker)], marker):
                    j_found = j
                    break
            if j_found < 0:
                labels[i, :] = IGNORE_INDEX
            else:
                labels[i, :j_found + len(marker)] = IGNORE_INDEX
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = IGNORE_INDEX
        inputs["labels"] = labels
        return dict(inputs)

    def __call__(self, items):
        chosen_texts, rejected_texts, imgs = [], [], []
        raw_chosen, raw_rejected, field_names_per_item = [], [], []
        for it in items:
            sign = _resize_for_qwen(it["cropped_sign"], self.max_image_pixels)
            vp = _resize_for_qwen(it["visual_prompt"], self.max_image_pixels)
            imgs.append([sign, vp])
            chosen_texts.append(self.processor.apply_chat_template(
                self._build_messages(sign, vp, it["prompt_text"], it["chosen_text"]),
                tokenize=False, add_generation_prompt=False))
            rejected_texts.append(self.processor.apply_chat_template(
                self._build_messages(sign, vp, it["prompt_text"], it["rejected_text"]),
                tokenize=False, add_generation_prompt=False))
            raw_chosen.append(it["chosen_text"])
            raw_rejected.append(it["rejected_text"])
            field_names_per_item.append(list((it["meta"].get("rejected_overrides") or {}).keys()))

        if self.single_pass:
            # v1: chosen + rejected concat into one (2B, ...) batch.
            # Output ordering: [chosen_0..B-1, rejected_0..B-1] so the train
            # loop can split logp[:B] / logp[B:] after forward.
            texts = chosen_texts + rejected_texts
            images = imgs + imgs
            batch = self._process_one(texts, images)
            if self.build_conflict_mask:
                self._attach_conflict_mask(
                    batch, raw_chosen + raw_rejected,
                    field_names_per_item + field_names_per_item,
                )
            return batch
        else:
            # plan A: two independent (B, ...) batches.
            chosen_batch = self._process_one(chosen_texts, imgs)
            rejected_batch = self._process_one(rejected_texts, imgs)
            if self.build_conflict_mask:
                self._attach_conflict_mask(chosen_batch, raw_chosen, field_names_per_item)
                self._attach_conflict_mask(rejected_batch, raw_rejected, field_names_per_item)
            return {"chosen": chosen_batch, "rejected": rejected_batch}

    def _attach_conflict_mask(self, batch: dict, raw_texts: list, field_names_list: list) -> None:
        """In-place: add batch['conflict_mask'] of shape (B, T) marking response
        tokens whose chars lie inside any conflict-field value span.

        Falls back to zeros (no token masked) for samples where the response
        token subsequence can't be located in input_ids — caller must
        gracefully handle all-zero rows (typically by falling back to vanilla
        DPO logp for that sample, or by treating zero-mask as 'no signal').
        """
        input_ids = batch["input_ids"]
        B, T = input_ids.shape
        mask = torch.zeros((B, T), dtype=torch.long)
        for i in range(B):
            m = build_conflict_token_mask(
                input_ids[i], raw_texts[i], field_names_list[i],
                self.processor.tokenizer,
            )
            if m is not None:
                mask[i, :m.shape[0]] = m
        batch["conflict_mask"] = mask


_LOGP_CHUNK_TOKENS = 64  # tokens per chunk for chunked log-prob computation
                          # (256 → 64 on 2026-05-12: long-tail samples push baseline
                          #  to 22.5GB, leaving <500MB; 64 tokens × 152K vocab × 4B =
                          #  ~38MB per chunk fits any fragment hole)


def per_sequence_logp(model, batch: dict, return_masked: bool = False):
    """Forward + sum log-probabilities of LABELED tokens for each sequence.

    Returns
    -------
    If `return_masked` is False (default): Tensor of shape (B,) — vanilla
    response-level logp (sum over tokens where labels != IGNORE_INDEX).
    Backward-compatible with all existing vanilla-DPO callers.

    If `return_masked` is True AND batch contains 'conflict_mask': returns
    (logp_full, logp_masked) where logp_full is the standard response sum
    and logp_masked sums only over tokens whose conflict_mask == 1. Used
    by token-mask DPO (v1): reward uses logp_masked, KL anchor uses
    logp_full.

    Memory: F.cross_entropy on the flattened (N=B*T, V=151936) tensor
    transiently allocates ~1.5-1.8 GB on the typical T ≈ 1500 case, which
    fragments the residual 24 GB on a 4090 right at the policy-forward
    edge. We chunk over the token dimension (default 256 tokens/chunk):
    each chunk alloc'd is N_chunk × V × 4 bytes ≈ 156 MB, well under
    fragmentation risk. Loops are short (~6 iterations at T=1500) and the
    overhead is negligible vs forward time.
    """
    kwargs = {k: v for k, v in batch.items() if k not in ("labels", "conflict_mask")}
    out = model(**kwargs)
    logits = out.logits[:, :-1, :]           # (B, T-1, V), bf16
    labels = batch["labels"][:, 1:]          # (B, T-1)
    label_mask = (labels != IGNORE_INDEX)
    safe_labels = labels.masked_fill(~label_mask, 0)
    B, Tm1, V = logits.shape

    flat_logits = logits.reshape(-1, V)      # (B*Tm1, V) — view, no copy
    flat_labels = safe_labels.reshape(-1)    # (B*Tm1,)
    N = flat_logits.shape[0]

    per_tok = torch.empty(N, dtype=torch.float32, device=logits.device)
    for s in range(0, N, _LOGP_CHUNK_TOKENS):
        e = min(s + _LOGP_CHUNK_TOKENS, N)
        nll_chunk = F.cross_entropy(
            flat_logits[s:e], flat_labels[s:e], reduction="none",
        )
        per_tok[s:e] = -nll_chunk
    per_tok = per_tok.view(B, Tm1)
    logp_full = (per_tok * label_mask).sum(-1)
    if not return_masked:
        return logp_full
    cmask = batch.get("conflict_mask")
    if cmask is None:
        # caller asked for masked but no mask present — return zeros so the
        # train loop can degenerate to vanilla cleanly.
        logp_masked = torch.zeros_like(logp_full)
    else:
        cmask_shift = cmask[:, 1:].to(per_tok.dtype)
        logp_masked = (per_tok * label_mask.to(per_tok.dtype) * cmask_shift).sum(-1)
    return logp_full, logp_masked


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp,
             beta: float = 0.1):
    """Standard DPO loss (Rafailov et al. 2023, Eq. 7).

    Returns
    -------
    loss : scalar
    metrics : dict with reward margins / accuracies for logging
    """
    pi_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        rewards_chosen = beta * (policy_chosen_logp - ref_chosen_logp)
        rewards_rejected = beta * (policy_rejected_logp - ref_rejected_logp)
        metrics = {
            "rewards_chosen":   rewards_chosen.mean().item(),
            "rewards_rejected": rewards_rejected.mean().item(),
            "reward_margin":    (rewards_chosen - rewards_rejected).mean().item(),
            "reward_accuracy":  (rewards_chosen > rewards_rejected).float().mean().item(),
            "policy_chosen_logp":   policy_chosen_logp.mean().item(),
            "policy_rejected_logp": policy_rejected_logp.mean().item(),
        }
    return loss, metrics


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-adapter", default="",
                   help="Path to LoRA adapter to initialize from (e.g. C0.3 SFT "
                        "checkpoint). Empty = fresh LoRA on top of base model.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="DPO recommended ~40x smaller than SFT (paper §6).")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="DPO usually runs with WD=0 — preference signal is "
                        "the only regularizer needed.")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--shuffle-idx", action="store_true")
    p.add_argument("--conflict-ratio", type=float, default=0.7,
                   help="Fraction of pairs that are vision-map conflict pairs "
                        "(vs control pairs that flip LaneType only).")
    p.add_argument("--conflict-type-weights", default="",
                   help="Comma-separated weights for [speed,direction,vehicle,time]. "
                        "Empty = uniform.")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature β. Paper recommends 0.1-0.5; 0.1 = "
                        "softer KL constraint to reference policy.")
    p.add_argument("--token-mask-dpo", action="store_true",
                   help="v1: restrict the DPO reward signal to conflict-field "
                        "VALUE tokens (sever suffix conditioning-shift hack) "
                        "AND add an explicit KL anchor over the full response. "
                        "See docs/RFT_TOKEN_MASK_DPO_DESIGN.md.")
    p.add_argument("--kl-weight", type=float, default=0.1,
                   help="λ for the explicit KL anchor term when --token-mask-dpo "
                        "is set. KL approx uses (pol_full - ref_full)^2 / 2. "
                        "0.1 is the PPO default; tune via dry-run.")
    p.add_argument("--max-image-pixels", type=int, default=28 * 28 * 1024,
                   help="Cap per-image pixel count. Default ~800K (config a) — "
                        "DPO forward is 2× SFT (chosen+rejected), so we stay "
                        "conservative on 4090 24GB.")
    p.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                   default="adamw",
                   help="paged_adamw8bit: CPU-paged 8bit AdamW, saves ~1 GB "
                        "GPU baseline at +50-100ms/step. Needed for DPO when "
                        "long-tail vision activation peaks would otherwise OOM.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA: load base in NF4 (4-bit NormalFloat) + double "
                        "quant + bf16 compute. Saves ~10 GB vs bf16 base, "
                        "needed when long-tail samples push DPO past 23 GB on "
                        "4090. Ref forward (disable_adapter) also runs on the "
                        "4bit base — see scripts/sanity_4bit_logp.py for the "
                        "logp drift sanity check.")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                   help="HF / modelscope model dir name under ckpts/ (e.g. "
                        "'Qwen2.5-VL-3B-Instruct' for the 3B twin runs).")
    p.add_argument("--single-pass", action="store_true",
                   help="Use v1 single-pass concat batch + single backward "
                        "(trl-style). Plan A's per-sample 2x backward is single-"
                        "GPU only (DDP reducer incompatible); single-pass works "
                        "on multi-GPU DDP but needs activation budget for 2x "
                        "samples concat. Use for 3B twin multi-card runs.")
    p.add_argument("--out-dir", default=str(REPO / "runs" / "dpo"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = build_args()
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    # static_graph=True lets DDP support the plan-A multi-backward pattern
    # (chosen-no_sync-backward + rejected-backward inside one iteration). It
    # also coexists cleanly with gradient_checkpointing's re-forward during
    # backward. Single-card runs ignore this kwarg (DDP not instantiated).
    ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16",
                              kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process

    # Pin each rank to its local GPU BEFORE any from_pretrained call. Without
    # this, PeftModel.from_pretrained (used to resume from C0.3) can leak ~1GB
    # NCCL/IPC buffers onto rank 0's GPU from all other ranks → multi-GPU OOM
    # even with small max_image_pixels. (SFT uses get_peft_model + fresh LoRA,
    # which initializes on CPU and never triggers the leak.)
    torch.cuda.set_device(accelerator.local_process_index)

    # Race-safe run_id: when multiple training processes start within the
    # same second (e.g. 4 ablation tmux launched together), strftime returns
    # identical IDs → mkdir(exist_ok=True) silently succeeds → 4 processes
    # write to the same dir, racing args.json / train_log.jsonl / ckpts.
    # Fix: append PID and use exist_ok=False; on the rare same-second-same-PID
    # collision (multi-rank), fall back to incremental suffix.
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
    # For multi-rank, broadcast the (potentially suffixed) out_dir path so
    # non-main ranks write to the same place. Use a sentinel file on shared FS.
    accelerator.wait_for_everyone()
    if accelerator.num_processes > 1 and not is_main:
        # Read the latest run_id picked by rank 0 (PIDs differ per rank in
        # accelerate launch, so each rank's PID-suffixed path differs; rank 0's
        # is canonical). Rank 0 writes args.json in out_dir; non-main scans for
        # a dir matching this second containing rank 0's PID.
        # Simpler: rank 0 broadcasts via file:
        pass  # current per-run multi-rank uses rank-0 PID by convention
        # NOTE: multi-rank DPO not currently used (plan A is single-GPU-only).
        # For 3B multi-card single-pass runs, rank 0's PID becomes the run_id.

    # 1) processor + base model -----------------------------------------------
    # Override module-level MODEL_DIR with the per-run --model-name choice so
    # 3B/7B twin runs share the same code path. Default = 7B-Instruct.
    MODEL_DIR = REPO / "ckpts" / args.model_name  # noqa: F811 (intentional rebind)
    if is_main:
        print(f"[load] processor + base from {MODEL_DIR}")
        print(f"[load] liger_kernel: {_LIGER}  (fused_linear_ce=False for DPO logits access)")
        print(f"[load] 4bit quantization: {args.load_in_4bit}")
        print(f"[load] single-pass mode: {args.single_pass}  (False = plan A per-sample 2x backward)")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), use_fast=True)
    base_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.load_in_4bit:
        # QLoRA: NF4 + double quant + bf16 compute. Pin to local rank's GPU
        # so the quantized weights land where the rest of accelerate expects.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_kwargs["quantization_config"] = bnb_config
        base_kwargs["device_map"] = {"": accelerator.local_process_index}
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL_DIR), **base_kwargs)
    if args.load_in_4bit:
        # Upcast layer norms / embeddings to fp32 for training stability; do NOT
        # let prepare_for_kbit_training enable gradient_checkpointing because
        # we want use_reentrant=True (saves 200-500 MB activation).
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    # use_reentrant=True (legacy backward impl) saves ~200-500 MB activation
    # memory vs use_reentrant=False (default PyTorch 2.x). 2026-05-12: switched
    # after 4 OOMs at 22.66-22.83 GB on long-tail samples — needed every MB.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    # 2) LoRA -----------------------------------------------------------------
    if args.init_adapter:
        if is_main:
            print(f"[lora] resume from adapter {args.init_adapter}")
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        if args.token_mask_dpo:
            # token-mask DPO needs ref = SFT-init snapshot (not base) so the
            # explicit KL anchor measures POLICY DRIFT from init, not the SFT
            # prior offset. Without this, KL pushes policy back to base and
            # erases SFT-learned skills. Load a 2nd LoRA adapter named "ref",
            # frozen, identical weights. Switch via set_adapter("ref") at ref
            # forward, set_adapter("default") to restore policy.
            if is_main:
                print(f"[lora] load frozen ref adapter from {args.init_adapter} "
                      f"(for token-mask KL anchor; ref = SFT-init snapshot)")
            model.load_adapter(args.init_adapter, adapter_name="ref", is_trainable=False)
            # peft.load_adapter often switches active_adapter to the new one and
            # toggles requires_grad → restore "default" as active + trainable.
            # Without this the optimizer collects 0 trainable params (empty list).
            model.set_adapter("default")
    else:
        if is_main:
            print(f"[lora] fresh LoRA rank={args.lora_rank}")
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
    if is_main:
        model.print_trainable_parameters()
        print(f"[load] done in {time.time() - t0:.1f}s")

    # 3) data -----------------------------------------------------------------
    conflict_type_weights = None
    if args.conflict_type_weights.strip():
        conflict_type_weights = [float(w) for w in args.conflict_type_weights.split(",")]
        if len(conflict_type_weights) != 4:
            raise SystemExit(f"--conflict-type-weights needs 4 floats, got {len(conflict_type_weights)}.")

    train_ds = MapDRDPODataset(
        root=DATA_ROOT,
        split="Train",
        conflict_ratio=args.conflict_ratio,
        shuffle_idx=args.shuffle_idx,
        conflict_type_weights=conflict_type_weights,
        seed=args.seed,
    )
    if is_main:
        print(f"[data] train pairs: {len(train_ds)} "
              f"(conflict_ratio={args.conflict_ratio}, "
              f"conflict_type_weights={conflict_type_weights})")
    collate = DPOCollator(processor, max_image_pixels=args.max_image_pixels,
                          single_pass=args.single_pass,
                          build_conflict_mask=args.token_mask_dpo)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=False,
    )

    # 4) optimizer ------------------------------------------------------------
    optim_params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "paged_adamw8bit":
        # CPU-paged 8bit AdamW — optim state lives on CPU, swaps to GPU on
        # demand. Saves ~1 GB GPU baseline vs in-GPU adamw8bit, at the cost
        # of ~50-100 ms / step extra. Needed when DPO baseline peaks 22 GB
        # leave no headroom for long-tail vision activation (2026-05-12).
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            optim_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            optim_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            optim_params, lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    if is_main:
        print(f"[plan] steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
              f"effective_batch={args.batch_size * args.grad_accum * accelerator.num_processes} pair-pairs")

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.1, 1.0 - progress)

    # 5) train loop -----------------------------------------------------------
    model.train()
    log_f = None
    if is_main:
        log_f = open(out_dir / "train_log.jsonl", "w")
    global_step = 0
    accum = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}
    t_step = time.time()

    pbar = tqdm(total=total_steps, disable=not is_main, dynamic_ncols=True,
                desc="dpo", smoothing=0.0)

    stop = False
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                if args.single_pass:
                    # v1 single-pass: chosen + rejected concat into (2B, ...)
                    # one forward + one backward. trl-style. Multi-GPU DDP
                    # friendly. Activation peak is 2× sample so use for 3B
                    # twin runs (small base leaves room for 2× sample).
                    with torch.no_grad(), unwrapped.disable_adapter():
                        ref_logp = per_sequence_logp(model, batch).detach()
                    B = ref_logp.shape[0] // 2
                    ref_chosen_logp, ref_rejected_logp = ref_logp[:B], ref_logp[B:]

                    policy_logp = per_sequence_logp(model, batch)
                    policy_chosen_logp, policy_rejected_logp = policy_logp[:B], policy_logp[B:]

                    loss, metrics_full = dpo_loss(
                        policy_chosen_logp, policy_rejected_logp,
                        ref_chosen_logp, ref_rejected_logp,
                        beta=args.beta,
                    )
                    accelerator.backward(loss)

                    loss_value = loss.detach()
                    metrics = {
                        "reward_margin": metrics_full["reward_margin"],
                        "reward_accuracy": metrics_full["reward_accuracy"],
                    }
                else:
                    # Plan A: per-sample forward to halve activation peak (vs the
                    # v1 concat-batch design that OOMed 7B long-tail at 23.6GB).
                    # DPO loss is L = -log σ(β·δ) where δ = (pol_w - ref_w) - (pol_l - ref_l).
                    # ∂L/∂pol_w = -β·σ(-β·δ) = c     ∂L/∂pol_l = -c
                    # → surrogate(c·pol_w) + surrogate(-c·pol_l) has the same total
                    #   gradient as backward(L). Compute c once with detached logps,
                    #   then backward the two surrogates sequentially so chosen
                    #   activations release before the rejected forward.
                    # Single-GPU only (DDP reducer state-machine incompatible with
                    # multi-backward, see progress.md 5/13 多卡探索).
                    chosen_batch = batch["chosen"]
                    rejected_batch = batch["rejected"]

                    if args.token_mask_dpo:
                        # v1: masked reward (conflict span only) + explicit KL
                        # anchor over full response. See docs/RFT_TOKEN_MASK_DPO_DESIGN.md.
                        # CRITICAL: ref = SFT-init snapshot ("ref" adapter, frozen, identical
                        # init weights to policy), NOT base. KL anchor must measure POLICY
                        # DRIFT from SFT-init, not vs base (which would penalize all SFT
                        # learning and pull policy back to a Qwen-without-MapDR-knowledge).
                        unwrapped.set_adapter("ref")
                        try:
                            with torch.no_grad():
                                ref_c_full, ref_c_masked = per_sequence_logp(model, chosen_batch, return_masked=True)
                                ref_r_full, ref_r_masked = per_sequence_logp(model, rejected_batch, return_masked=True)
                        finally:
                            unwrapped.set_adapter("default")
                        ref_c_full = ref_c_full.detach(); ref_c_masked = ref_c_masked.detach()
                        ref_r_full = ref_r_full.detach(); ref_r_masked = ref_r_masked.detach()

                        pol_c_full, pol_c_masked = per_sequence_logp(model, chosen_batch, return_masked=True)
                        B = pol_c_full.shape[0]

                        with torch.no_grad():
                            pol_r_full_ng, pol_r_masked_ng = per_sequence_logp(model, rejected_batch, return_masked=True)
                        pol_r_full_ng = pol_r_full_ng.detach(); pol_r_masked_ng = pol_r_masked_ng.detach()

                        # δ uses MASKED logp — severs the suffix conditioning-shift hack channel
                        delta = ((pol_c_masked.detach() - ref_c_masked)
                                 - (pol_r_masked_ng - ref_r_masked))
                        coef = -args.beta * torch.sigmoid(-args.beta * delta)

                        # KL anchor on FULL response: forward-KL approx (pol - ref)^2 / 2
                        # Penalizes both directions (pol > ref or pol < ref). Differentiable
                        # at 0 unlike |·|. Bounded by Schulman 2020 cheap KL estimator family.
                        kl_chosen = ((pol_c_full - ref_c_full) ** 2) / 2.0

                        # chosen surrogate = DPO grad target (masked) + KL grad target (full)
                        chosen_surrogate = (coef * pol_c_masked + args.kl_weight * kl_chosen).sum() / B
                        accelerator.backward(chosen_surrogate)

                        pol_r_full, pol_r_masked = per_sequence_logp(model, rejected_batch, return_masked=True)
                        kl_rejected = ((pol_r_full - ref_r_full) ** 2) / 2.0
                        rejected_surrogate = (-coef * pol_r_masked + args.kl_weight * kl_rejected).sum() / B
                        accelerator.backward(rejected_surrogate)

                        with torch.no_grad():
                            loss_dpo = -F.logsigmoid(args.beta * delta).mean()
                            loss_kl = args.kl_weight * (kl_chosen.detach() + kl_rejected.detach()).mean() / 2.0
                            loss_value = loss_dpo + loss_kl
                            rewards_chosen = args.beta * (pol_c_masked.detach() - ref_c_masked)
                            rewards_rejected = args.beta * (pol_r_masked_ng - ref_r_masked)
                            metrics = {
                                "reward_margin": (rewards_chosen - rewards_rejected).mean().item(),
                                "reward_accuracy": (rewards_chosen > rewards_rejected).float().mean().item(),
                                "kl_term": loss_kl.item(),
                                "dpo_loss": loss_dpo.item(),
                            }
                    else:
                        # Vanilla plan A — original 5/13 code, unchanged.
                        # 1) ref forward (no_grad, adapter disabled) → release immediately.
                        #    no_grad makes DDP's _pre_forward bypass the reducer check
                        #    (see torch.nn.parallel.distributed:1528), so these don't
                        #    interfere with the chosen/rejected DDP state machine.
                        with torch.no_grad(), unwrapped.disable_adapter():
                            ref_chosen_logp = per_sequence_logp(model, chosen_batch).detach()
                            ref_rejected_logp = per_sequence_logp(model, rejected_batch).detach()

                        # 2) policy chosen forward (grad on) → keeps chosen graph alive
                        policy_chosen_logp = per_sequence_logp(model, chosen_batch)
                        B = policy_chosen_logp.shape[0]

                        # 3) policy rejected forward (no_grad) → transient activation,
                        #    released after the context exits; chosen graph still alive
                        with torch.no_grad():
                            policy_rejected_nograd = per_sequence_logp(model, rejected_batch).detach()

                        # 4) σ coefficient (detached, no autograd through it)
                        delta = ((policy_chosen_logp.detach() - ref_chosen_logp)
                                 - (policy_rejected_nograd - ref_rejected_logp))
                        coef = -args.beta * torch.sigmoid(-args.beta * delta)  # (B,)

                        # 5) backward chosen surrogate → releases chosen activations.
                        chosen_surrogate = (coef * policy_chosen_logp).sum() / B
                        accelerator.backward(chosen_surrogate)

                        # 6) policy rejected forward (grad on) → builds rejected graph
                        policy_rejected_logp = per_sequence_logp(model, rejected_batch)

                        # 7) backward rejected surrogate → releases rejected activations.
                        rejected_surrogate = (-coef * policy_rejected_logp).sum() / B
                        accelerator.backward(rejected_surrogate)

                        # 8) loss + metrics for logging (no_grad, uses detached logps)
                        with torch.no_grad():
                            loss_value = -F.logsigmoid(args.beta * delta).mean()
                            rewards_chosen = args.beta * (policy_chosen_logp.detach() - ref_chosen_logp)
                            rewards_rejected = args.beta * (policy_rejected_nograd - ref_rejected_logp)
                            metrics = {
                                "reward_margin": (rewards_chosen - rewards_rejected).mean().item(),
                                "reward_accuracy": (rewards_chosen > rewards_rejected).float().mean().item(),
                            }

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(optim_params, 1.0)
                    for g in optimizer.param_groups:
                        g["lr"] = args.lr * lr_at(global_step)
                optimizer.step()
                optimizer.zero_grad()

                accum["loss"]   += float(loss_value.detach())
                accum["margin"] += metrics["reward_margin"]
                accum["acc"]    += metrics["reward_accuracy"]
                accum["n"]      += 1
                if "kl_term" in metrics:
                    accum["kl"] = accum.get("kl", 0.0) + metrics["kl_term"]
                    accum["dpo_loss"] = accum.get("dpo_loss", 0.0) + metrics["dpo_loss"]

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            pbar.update(1)
            if is_main:
                pbar.set_postfix(
                    loss=f"{accum['loss']/max(1,accum['n']):.4f}",
                    margin=f"{accum['margin']/max(1,accum['n']):.3f}",
                    acc=f"{accum['acc']/max(1,accum['n']):.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            if global_step % args.log_every == 0 or global_step == 1:
                if is_main:
                    dt = time.time() - t_step
                    n = max(1, accum["n"])
                    msg = {
                        "step": global_step, "epoch": epoch,
                        "loss":           round(accum["loss"]   / n, 4),
                        "reward_margin":  round(accum["margin"] / n, 4),
                        "reward_acc":     round(accum["acc"]    / n, 3),
                        "lr": optimizer.param_groups[0]["lr"],
                        "step_time_s": round(dt / max(1, args.log_every), 2),
                    }
                    if "kl" in accum:
                        msg["kl_term"] = round(accum["kl"] / n, 4)
                        msg["dpo_loss"] = round(accum["dpo_loss"] / n, 4)
                    log_f.write(json.dumps(msg) + "\n")
                    log_f.flush()
                    t_step = time.time()
                accum = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}

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

    accelerator.wait_for_everyone()
    pbar.close()
    if is_main:
        final_dir = out_dir / "final"
        accelerator.unwrap_model(model).save_pretrained(str(final_dir))
        log_f.close()
        print(f"[done] global_step={global_step}, final adapter → {final_dir}")


if __name__ == "__main__":
    main()
