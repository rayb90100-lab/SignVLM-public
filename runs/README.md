# `runs/` — training outputs

Training and evaluation scripts write their outputs under here:

```
runs/sft/<timestamp>/final/        SFT LoRA adapter (used by eval / DPO init)
runs/sft/<timestamp>/eval_<...>/   eval_sft.py output (preds_shard*.jsonl + metrics)
runs/dpo/<timestamp>/...           DPO outputs (experimental, results withheld)
runs/grpo/<timestamp>/...          GRPO outputs (experimental, results withheld)
```

This directory is created on demand and its contents are ignored by
`.gitignore`. To free disk: `rm -rf runs/*` (you'll keep this README via the
`!runs/README.md` rule).
