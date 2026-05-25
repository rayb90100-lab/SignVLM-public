# `ckpts/` — base model checkpoints

Put (or symlink) the base VLM checkpoint here, named exactly to match the
`MODEL_NAME` env var used by the training scripts (default
`Qwen2.5-VL-7B-Instruct`):

```bash
# Option A: download directly
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir ckpts/Qwen2.5-VL-7B-Instruct

# Option B: symlink to a copy that already exists somewhere on disk
ln -s /your/absolute/path/to/Qwen2.5-VL-7B-Instruct ckpts/Qwen2.5-VL-7B-Instruct
```

Use `Qwen2.5-VL-3B-Instruct` instead if you want to run the smaller-size
twin experiments (set `MODEL_NAME=Qwen2.5-VL-3B-Instruct` when launching).

See the top-level README §3 for the full setup recipe.

> The contents of this directory (other than this README) are ignored by
> `.gitignore` and should never be committed.
