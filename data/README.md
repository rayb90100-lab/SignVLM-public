# `data/` — dataset directory

Put your **MapDR** dataset here, named `MapDR/`. A symlink to wherever the
dataset actually lives is the recommended pattern (the dataset is ~10 GB+
and shouldn't live inside your code checkout):

```bash
# from the repo root
ln -s /your/absolute/path/to/MapDR data/MapDR
```

Expected layout after the symlink resolves:

```
data/MapDR/
  <scene_id>/
    data.json
    label.json
    img/*.jpg
  ...
  split.json          # Train/Test scene-id lists
```

For the optional `render_visual_prompts.py` sanity-check script, also place
a handful (~8) of MapDR scenes under `data/zeroshot_demo/`.

See the top-level README §3 for the full data-preparation recipe.

> The contents of this directory (other than this README) are ignored by
> `.gitignore` and should never be committed.
