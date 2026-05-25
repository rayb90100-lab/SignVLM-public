"""SFT eval: run a trained LoRA adapter over MapDR Test split and report
the 5 RuleVLM metrics + lane_correctness extension.

Single-script: this file is both the launcher and the worker.
  - Launcher mode (--gpus 1,2,3,4,5,6 or default): forks one subprocess per GPU,
    each handling its strided slice of Test scenes. Shard-0 stdout is passed
    through to the current terminal (so you see tqdm live), other shards are
    teed into shard{i}.log. After all shards finish, this process merges the
    per-shard preds_shard*.jsonl into one metrics_merged.json.
  - Worker mode (--worker): runs a single shard; CUDA_VISIBLE_DEVICES set by
    the launcher restricts it to one GPU. Internal use only.

Pipeline (mirrors zeroshot_dualimg.py + train_sft.py message format):
  1. load Qwen2.5-VL base (bf16, flash_attention_2) + PEFT adapter
  2. MapDRDataset(split='Test', shuffle_idx=False, map_perturb_prob=0.0)
  3. for each scene: build the same 3-turn prompt as training (system + user
     [sign, vp, prompt_text]) with add_generation_prompt=True, generate with
     do_sample=False, decode the new tokens.
  4. accumulate via MapDRMetric.add(pred_text, gt_text), write per-scene jsonl.

Usage:
  python scripts/eval_sft.py --adapter runs/sft/<id>/final
       # default 6-card launcher: GPUs 1,2,3,4,5,6 (GPU 0 left free)

  python scripts/eval_sft.py --adapter runs/sft/<id>/final --gpus 0
       # single-card

  python scripts/eval_sft.py --adapter runs/sft/<id>/final --gpus 1,2 --limit 60
       # 2-card smoke
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Worker-only heavy imports (skip in launcher to keep startup fast)
_IS_WORKER = "--worker" in sys.argv

if _IS_WORKER:
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    sys.path.insert(0, str(REPO / "src"))
    from dataset import MapDRDataset  # noqa: E402
    from metric import MapDRMetric  # noqa: E402
else:
    sys.path.insert(0, str(REPO / "src"))
    from metric import compute_metrics  # noqa: E402

DEFAULT_MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # legacy default for 7B runs without args.json model_name
MODEL_DIR = REPO / "ckpts" / DEFAULT_MODEL_NAME  # module-level default; launcher/worker may rebind via args.model_name
DATA_ROOT = REPO / "data" / "MapDR"

# Must match scripts/train_sft.py SYSTEM_PROMPT exactly — train/eval prompt parity.
SYSTEM_PROMPT = (
    "You are a sign-aware autonomous-driving assistant. "
    "Read the traffic sign and the rendered front view, then output a JSON dict "
    "with rules / lane_assignment / plan as instructed."
)


def _resize_for_qwen(img, max_pixels: int):
    """Same as train_sft.py: round both sides to multiples of 28, cap pixel count."""
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


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="",
                   help="LoRA adapter dir (e.g. runs/sft/<run_id>/final). "
                        "If empty + --no-adapter, run zero-shot base model")
    p.add_argument("--no-adapter", action="store_true",
                   help="Zero-shot eval: skip PEFT, run base model only. "
                        "Use with --model-name to pick backbone (3B / 7B)")
    p.add_argument("--gpus", default="1,2,3,4,5,6",
                   help="Comma-separated CUDA device ids for the launcher to fork "
                        "across. Default leaves GPU 0 free. Use a single id for "
                        "single-card eval.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-image-pixels", type=int, default=0,
                   help="Cap per-image pixel count. 0 (default) = auto-read from "
                        "<adapter>/../args.json so eval matches training; "
                        "set explicitly to override.")
    p.add_argument("--model-name", default="",
                   help="Base model name under ckpts/ (e.g. Qwen2.5-VL-7B-Instruct, "
                        "Qwen2.5-VL-3B-Instruct). Empty (default) = auto-read from "
                        "<adapter>/../args.json model_name field; legacy 7B runs "
                        "without that field fall back to Qwen2.5-VL-7B-Instruct.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only eval first N scenes per shard (smoke test).")
    p.add_argument("--map-perturb", type=float, default=0.0,
                   help="LEGACY (Task 6.9 ablation): per-scene HD-Map perturbation "
                        "probability under perturb_mode='noise'. 0 (default) = clean "
                        "eval. 1.0 = every scene perturbed (LaneDirection flip / "
                        "speed change / centerline swap, input+GT moved together). "
                        "Measures input-fidelity collapse, NOT vision-faithfulness. "
                        "Mutually exclusive with --eval-mode conflict.")
    p.add_argument("--n-conflict-fields", type=int, default=1,
                   help="multi-field ablation (eval-only): n distinct conflict types per scene, 1-4. "
                        "1 = default single-field, 2-4 = multi-field ablation")
    p.add_argument("--eval-mode", choices=("standard", "conflict"), default="standard",
                   help="standard (default) = clean or noise-perturbed eval per "
                        "--map-perturb. conflict = CAVP main eval: every test scene "
                        "gets one panel-field conflict perturbation (perturb_prob "
                        "internally fixed to 1.0); GT untouched; metric reports "
                        "conflict_resolution_acc + per-type acc.")
    p.add_argument("--merge-adapter", action="store_true",
                   help="Call merge_and_unload() on the adapter for faster inference.")
    p.add_argument("--out-dir", default="",
                   help="Override output dir; default = <adapter>/../eval_<timestamp>")
    p.add_argument("--seed", type=int, default=42)
    # Worker-internal flags (launcher sets these when forking)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Launcher mode: fork one subprocess per GPU, then merge.
# ---------------------------------------------------------------------------

def _resolve_max_pixels(args, adapter_dir):
    """If --max-image-pixels was 0/unset, read it from train run's args.json so
    eval geometry matches training. Train/eval mismatch silently degrades
    metrics (we caught run 081257 eval'd at 800K when trained at 1.6M)."""
    if args.max_image_pixels > 0:
        return args.max_image_pixels
    train_args_path = adapter_dir.parent / "args.json"
    if train_args_path.exists():
        train_args = json.loads(train_args_path.read_text())
        v = train_args.get("max_image_pixels", 802816)
        return int(v)
    return 802816  # legacy default


def _resolve_model_name(args, adapter_dir):
    """If --model-name was empty, read it from train run's args.json. Legacy 7B
    runs that predate the model-name flag won't have the field — fall back to
    Qwen2.5-VL-7B-Instruct. Same pattern as _resolve_max_pixels: avoid silent
    train/eval mismatch (3B adapter on 7B base raises LoRA shape errors)."""
    if args.model_name:
        return args.model_name
    train_args_path = adapter_dir.parent / "args.json"
    if train_args_path.exists():
        train_args = json.loads(train_args_path.read_text())
        return train_args.get("model_name", DEFAULT_MODEL_NAME)
    return DEFAULT_MODEL_NAME


def run_launcher(args):
    if args.no_adapter:
        if not args.model_name:
            raise SystemExit("--no-adapter requires --model-name (e.g., Qwen2.5-VL-7B-Instruct)")
        if not args.out_dir:
            raise SystemExit("--no-adapter requires explicit --out-dir")
        adapter_dir = Path(args.out_dir).parent.resolve()  # for path resolution; not used to load adapter
        if args.max_image_pixels <= 0:
            args.max_image_pixels = 802816  # match SFT default for fair comparison
    else:
        adapter_dir = Path(args.adapter).resolve()
        if not adapter_dir.exists():
            raise SystemExit(f"adapter not found: {adapter_dir}")
        args.max_image_pixels = _resolve_max_pixels(args, adapter_dir)
        args.model_name = _resolve_model_name(args, adapter_dir)
    model_dir = REPO / "ckpts" / args.model_name
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir} (resolved model_name={args.model_name!r})")

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_shards = len(gpu_ids)
    if n_shards == 0:
        raise SystemExit("--gpus is empty")

    # Sanity: parent CUDA_VISIBLE_DEVICES gets overridden by env["CUDA_VISIBLE_DEVICES"]=gpu
    # in the worker fork below. Without this warning, `CUDA_VISIBLE_DEVICES=6 python
    # eval_sft.py --gpus 0` silently runs on physical GPU 0, not GPU 6 (5/14 03:42 trap).
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        print(f"[launcher] NOTE: parent CUDA_VISIBLE_DEVICES='{parent_cvd}' "
              f"will be overridden — workers pinned to physical GPU id(s) {gpu_ids} "
              f"per --gpus (interpret as direct device ids, NOT positions within parent mask).")

    if args.eval_mode == "conflict" and args.map_perturb > 0:
        raise SystemExit(
            "--eval-mode conflict is mutually exclusive with --map-perturb "
            "(legacy noise eval). Pick one."
        )

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        # Tag the dir so clean / noisy / conflict eval don't collide
        if args.eval_mode == "conflict":
            suffix = "_conflict"
        elif args.map_perturb > 0:
            suffix = f"_p{args.map_perturb}".rstrip("0").rstrip(".")
        else:
            suffix = ""
        out_dir = adapter_dir.parent / f"eval_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[launcher] adapter   = {adapter_dir}")
    print(f"[launcher] model     = {args.model_name}  ({model_dir})")
    print(f"[launcher] eval_dir  = {out_dir}")
    print(f"[launcher] GPUs      = {','.join(gpu_ids)}  ({n_shards} shards)")
    print(f"[launcher] eval_mode = {args.eval_mode}, map_perturb= {args.map_perturb}")
    print(f"[launcher] max_pixels= {args.max_image_pixels}, max_new_tokens= {args.max_new_tokens}")
    print(f"[launcher] shard 0 streams to this terminal; shards 1+ → {out_dir}/shard{{i}}.log")
    print()

    procs, log_files = [], []
    for i, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--adapter", str(adapter_dir) if not args.no_adapter else "",
            "--model-name", args.model_name,
            "--shard-id", str(i),
            "--num-shards", str(n_shards),
            "--device", "cuda:0",
            "--out-dir", str(out_dir),
            "--max-image-pixels", str(args.max_image_pixels),
            "--max-new-tokens", str(args.max_new_tokens),
            "--map-perturb", str(args.map_perturb),
            "--eval-mode", args.eval_mode,
            "--seed", str(args.seed),
            "--n-conflict-fields", str(args.n_conflict_fields),
        ]
        if args.no_adapter:
            cmd += ["--no-adapter"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.merge_adapter:
            cmd += ["--merge-adapter"]

        if i == 0:
            # Pass shard 0 directly through to the current terminal so tqdm renders.
            p = subprocess.Popen(cmd, env=env)
            log_files.append(None)
        else:
            log_path = out_dir / f"shard{i}.log"
            lf = open(log_path, "w")
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
            log_files.append(lf)
        print(f"[launcher] start shard {i} on GPU {gpu} (pid={p.pid})")
        procs.append(p)

    # Forward SIGINT/SIGTERM to children so Ctrl+C kills the whole shard set
    # (subprocess.Popen does NOT propagate signals by default; without this the
    # launcher dies but workers keep running and hold GPU memory).
    def _kill_all(signum, frame):
        print(f"\n[launcher] received signal {signum}; killing {len(procs)} shards…")
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
        sys.exit(130)
    signal.signal(signal.SIGINT, _kill_all)
    signal.signal(signal.SIGTERM, _kill_all)

    print()
    print("[launcher] waiting for all shards…  (other shards: tail -f "
          f"{out_dir}/shard1.log)")
    print()

    failed = []
    for i, p in enumerate(procs):
        rc = p.wait()
        if log_files[i] is not None:
            log_files[i].close()
        if rc != 0:
            failed.append((i, rc))

    if failed:
        for i, rc in failed:
            print(f"[launcher] !! shard {i} failed with exit {rc}; see "
                  f"{out_dir}/shard{i}.log")
        raise SystemExit(1)

    # ---- merge ----
    print()
    print("[launcher] all shards done; merging…")
    shard_files = sorted(out_dir.glob("preds_shard*.jsonl"))
    preds, gts, scene_ids, conflict_metas = [], [], [], []
    for f in shard_files:
        n_before = len(preds)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            preds.append(r["pred_text"])
            gts.append(r["gt_text"])
            scene_ids.append(r["scene_id"])
            conflict_metas.append(r.get("conflict_meta"))
        print(f"  + {f.name}: {len(preds) - n_before} scenes")

    n_unique = len(set(scene_ids))
    if n_unique != len(scene_ids):
        print(f"[launcher] !! duplicate scene_ids: {len(scene_ids)} total, "
              f"{n_unique} unique (overlapping shards)")

    cm_arg = conflict_metas if args.eval_mode == "conflict" else None
    metrics = compute_metrics(preds, gts, conflict_metas=cm_arg)
    out_path = out_dir / "metrics_merged.json"
    out_path.write_text(json.dumps({
        "n_total_records": len(preds),
        "n_unique_scenes": n_unique,
        "n_shards_merged": len(shard_files),
        "adapter": str(adapter_dir),
        "eval_mode": args.eval_mode,
        "max_image_pixels": args.max_image_pixels,
        "max_new_tokens": args.max_new_tokens,
        "map_perturb": args.map_perturb,
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))

    print()
    print(f"[launcher] DONE — {len(preds)} records ({n_unique} unique) → {out_path}")
    print()
    print("=== Merged metrics ===")
    for k, v in metrics.items():
        if k == "_counts":
            continue
        print(f"  {k:18s} = {v:.4f}")
    c = metrics["_counts"]
    print(f"  parse_failures = {c['parse_failures']} / {c['total_scenes']}")
    print(f"  gt_rules / pairs = {c['gt_rules']} / {c['gt_pairs']}")
    print(f"  target_lane_present = {c['target_lane_present']}")


# ---------------------------------------------------------------------------
# Worker mode: run one shard.
# ---------------------------------------------------------------------------

def run_worker(args):
    if args.shard_id >= args.num_shards or args.shard_id < 0:
        raise SystemExit(f"--shard-id {args.shard_id} out of range for --num-shards {args.num_shards}")
    torch.set_float32_matmul_precision("high")

    if args.no_adapter:
        adapter_dir = Path(args.out_dir).parent.resolve() if args.out_dir else REPO
        if args.max_image_pixels == 0:
            args.max_image_pixels = 802816
    else:
        adapter_dir = Path(args.adapter).resolve()
        if not adapter_dir.exists():
            raise SystemExit(f"adapter not found: {adapter_dir}")
        # Worker may receive max_image_pixels / model_name from launcher (already
        # resolved); fall back to args.json read for direct-CLI worker invocation.
        if args.max_image_pixels == 0:
            args.max_image_pixels = _resolve_max_pixels(args, adapter_dir)
        if not args.model_name:
            args.model_name = _resolve_model_name(args, adapter_dir)
    global MODEL_DIR
    MODEL_DIR = REPO / "ckpts" / args.model_name
    if not MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {MODEL_DIR} (resolved model_name={args.model_name!r})")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        adapter_dir.parent / f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_lead = (args.shard_id == 0)
    if is_lead:
        print(f"[shard {args.shard_id}/{args.num_shards}] adapter = {adapter_dir}")
        print(f"[shard {args.shard_id}/{args.num_shards}] model   = {args.model_name} ({MODEL_DIR})")
        print(f"[shard {args.shard_id}/{args.num_shards}] out_dir = {out_dir}")
        print(f"[shard {args.shard_id}/{args.num_shards}] device = {args.device} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")

    print(f"[shard {args.shard_id}] load processor + base + adapter…")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), use_fast=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if args.no_adapter:
        print(f"[shard {args.shard_id}] ZERO-SHOT mode: skip PEFT, use base model only")
        model = base
    else:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        if args.merge_adapter:
            model = model.merge_and_unload()
    model = model.to(args.device)
    model.eval()
    print(f"[shard {args.shard_id}] load done in {time.time() - t0:.1f}s")

    if args.eval_mode == "conflict":
        # CAVP main eval: every test scene gets one conflict perturb; GT untouched.
        # --n-conflict-fields > 1 enables multi-field ablation (OOD wrt training)
        test_ds = MapDRDataset(
            root=DATA_ROOT,
            split="Test",
            shuffle_idx=False,
            perturb_mode="conflict",
            perturb_prob=1.0,
            n_conflict_fields=args.n_conflict_fields,
            seed=args.seed,
        )
    else:
        # Legacy standard eval: clean (map_perturb=0) or noise (map_perturb>0)
        test_ds = MapDRDataset(
            root=DATA_ROOT,
            split="Test",
            shuffle_idx=False,
            map_perturb_prob=args.map_perturb,
            seed=args.seed,
        )
    n_total = len(test_ds)
    indices = list(range(args.shard_id, n_total, args.num_shards))
    if args.limit > 0:
        indices = indices[: args.limit]
    print(f"[shard {args.shard_id}] test size = {n_total}; this shard = {len(indices)} scenes")

    metric = MapDRMetric()
    pred_log_path = out_dir / f"preds_shard{args.shard_id}.jsonl"
    pred_log = open(pred_log_path, "w")
    # smoothing=0.3 (tqdm default): s/it shows EWMA (recent / instantaneous)
    # while avg_s in postfix shows the cumulative mean — together they reveal
    # speed-up or slowdown trends across the shard.
    pbar = tqdm(indices, dynamic_ncols=True, smoothing=0.3,
                desc=f"shard{args.shard_id}",
                disable=not is_lead)  # only shard 0 renders tqdm

    n_parse_fail = 0
    t_loop = time.time()
    for i, ds_idx in enumerate(pbar):
        item = test_ds[ds_idx]
        sign = _resize_for_qwen(item["cropped_sign"], args.max_image_pixels)
        vp = _resize_for_qwen(item["visual_prompt"], args.max_image_pixels)

        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",
             "content": [
                 {"type": "image", "image": sign},
                 {"type": "image", "image": vp},
                 {"type": "text", "text": item["prompt_text"]},
             ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[[sign, vp]],
            padding=True,
            return_tensors="pt",
        ).to(args.device)

        t0 = time.time()
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        new_ids = out_ids[:, inputs.input_ids.shape[1]:]
        pred_text = processor.batch_decode(
            new_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        dt = time.time() - t0

        gt_text = item["gt_text"]
        conflict_meta = item["meta"].get("conflict_meta")
        metric.add(pred_text, gt_text, conflict_meta=conflict_meta)

        if metric.parse_failures > n_parse_fail:
            n_parse_fail = metric.parse_failures

        rec = {
            "scene_id": item["meta"]["scene_id"],
            "rep_ts": item["meta"]["rep_ts"],
            "n_centerlines": item["meta"]["n_centerlines"],
            "n_rules": item["meta"]["n_rules"],
            "abs_to_rel": item["meta"]["abs_to_rel"],
            "conflict_meta": conflict_meta,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "new_tokens": int(new_ids.shape[1]),
            "elapsed_s": round(dt, 2),
        }
        pred_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pred_log.flush()

        if is_lead:
            elapsed = time.time() - t_loop
            ok = (i + 1) - n_parse_fail
            pbar.set_postfix(
                parse_ok=f"{ok}/{i+1}",
                avg_s=f"{elapsed/(i+1):.1f}",
            )
    pbar.close()
    pred_log.close()

    final = metric.compute()
    metrics_path = out_dir / f"metrics_shard{args.shard_id}.json"
    metrics_path.write_text(json.dumps({
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "shard_size": len(indices),
        "limit": args.limit,
        "adapter": str(adapter_dir),
        "eval_mode": args.eval_mode,
        "max_image_pixels": args.max_image_pixels,
        "max_new_tokens": args.max_new_tokens,
        "map_perturb": args.map_perturb,
        "metrics": final,
    }, ensure_ascii=False, indent=2))
    print(f"[shard {args.shard_id}] {len(indices)} scenes in {time.time() - t_loop:.0f}s "
          f"→ {metrics_path}")


def main():
    args = build_args()
    if args.worker:
        run_worker(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
