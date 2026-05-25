"""Evaluate RuleVLM on MapDR test split (1076 scenes), output SignVLM-format predictions.

Two-phase:
  Phase 1: Generate RuleVLM-format QA data using their gen_data.py pipeline
           (this also renders RuleVLM's native visual_prompts + cropped_signs)
  Phase 2: Load RuleVLM model + chat() loop + save preds_shard0.jsonl

Output: directory with preds_shard0.jsonl + metrics_merged.json
Then run scripts/eval_canonical.py on the output dir to get v3 metrics.

Run via rulevlm conda env (which has transformers==4.42.3 etc).

Usage:
  conda activate rulevlm
  # Clone the official RuleVLM repo (https://github.com/MIV-XJTU/MapDR) and set
  # RULEVLM_REPO to point at it; default below is ../RuleVLM_ref/RuleVLM.
  RULEVLM_REPO=../RuleVLM_ref/RuleVLM \\
  python scripts/eval_rulevlm.py \\
      --mapdr-dir ./data \\
      --split-file ./data/MapDR/split.json \\
      --qa-dataset-dir /tmp/rulevlm_qa_data \\
      --adapter <path/to/rulevlm-adapter> \\
      --model-dir <path/to/qwen-vl-chat-base> \\
      --out-dir ./runs/rulevlm/eval_native \\
      [--gen-data-only] [--eval-only]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch

# Path to RuleVLM-original code so we can `from dataset.gen_data import ...`.
# Override with RULEVLM_REPO env var; default assumes the official repo is
# cloned to a sibling directory of this repo.
RULEVLM_REPO = os.environ.get('RULEVLM_REPO', str(Path(__file__).resolve().parent.parent.parent / 'RuleVLM_ref' / 'RuleVLM'))


def gen_qa_data(mapdr_dir: str, split_file: str, qa_dataset_dir: str, test_only: bool = True):
    """Generate RuleVLM-format QA data for our split.

    Monkey-patches gen_data globals + iterates test_uid_list only (default).
    """
    if RULEVLM_REPO not in sys.path:
        sys.path.insert(0, RULEVLM_REPO)
    # Need dataset to be a package
    init_file = Path(RULEVLM_REPO) / 'dataset' / '__init__.py'
    if not init_file.exists():
        init_file.touch()

    import dataset.gen_data as gd

    # Patch globals
    gd.mapdr_dataset_dir = mapdr_dir

    os.makedirs(qa_dataset_dir, exist_ok=True)
    visual_prompt_dir = os.path.join(qa_dataset_dir, 'visual_prompt')
    cropped_img_dir = os.path.join(qa_dataset_dir, 'cropped_img')  # NOTE: gen_data.py uses 'cropped_img', not 'cropped_sign'
    os.makedirs(visual_prompt_dir, exist_ok=True)
    os.makedirs(cropped_img_dir, exist_ok=True)
    gd.visual_prompt_dir = visual_prompt_dir
    gd.cropped_image_dir = cropped_img_dir  # CRITICAL: global referenced inside select_cropped_sign_image

    # Read split
    with open(split_file) as f:
        split = json.load(f)
    test_uids = split['Test']
    print(f'[gen] test_uids count: {len(test_uids)}')

    qa_list = []
    uid2ts = {}
    failed = []

    from tqdm import tqdm
    for i, uid in enumerate(tqdm(test_uids, desc='gen_qa')):
        try:
            image_dir = os.path.join(mapdr_dir, uid, 'img')
            data_file = os.path.join(mapdr_dir, uid, 'data.json')
            label_file = os.path.join(mapdr_dir, uid, 'label.json')

            target_ts, target_img_name, cropped_img_path = gd.select_cropped_sign_image(
                uid, image_dir, data_file
            )
            uid2ts[uid] = target_ts

            format_data, _ = gd.get_format_data(
                data_num=i,
                uid=uid,
                data_path=data_file,
                label_path=label_file,
                cropped_img_path=cropped_img_path,
                target_ts=target_ts,
                target_img_name=target_img_name,
                train_tmp_dir=None,
                is_shuffle=False,
            )
            qa_list.append(format_data)
        except Exception as e:
            failed.append({'uid': uid, 'err': str(e)})
            if len(failed) < 5:
                print(f'[gen] FAIL {uid}: {e}')

    print(f'[gen] success: {len(qa_list)}, failed: {len(failed)}')

    with open(os.path.join(qa_dataset_dir, 'test.json'), 'w') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    with open(os.path.join(qa_dataset_dir, 'uid2ts.json'), 'w') as f:
        json.dump(uid2ts, f)
    with open(os.path.join(qa_dataset_dir, 'failed.json'), 'w') as f:
        json.dump(failed, f, indent=2)
    print(f'[gen] wrote {qa_dataset_dir}/test.json ({len(qa_list)} samples), uid2ts.json, failed.json')


def parse_rulevlm_response_to_signvlm_schema(response: str) -> dict:
    """RuleVLM outputs `{rule_index: {attr_info: {...}, centerline: [...]}}` as Python-dict-like str.
    SignVLM uses `{rules: [{attr_info: {...}, centerline: [...]}, ...], plan: {...}}`.
    Convert RuleVLM → SignVLM shape (with empty plan since RuleVLM doesn't output it).
    """
    try:
        # Try JSON first, then Python literal (RuleVLM uses single quotes)
        try:
            rv_dict = json.loads(response)
        except Exception:
            try:
                rv_dict = json.loads(response.replace("'", '"'))
            except Exception:
                import ast
                rv_dict = ast.literal_eval(response)

        if not isinstance(rv_dict, dict):
            return {'rules': [], 'plan': {}, '_parse_err': 'not a dict'}

        rules = []
        for rule_idx, entry in rv_dict.items():
            if isinstance(entry, dict) and 'attr_info' in entry:
                rule = {'attr_info': entry['attr_info'], 'centerline': entry.get('centerline', [])}
                rules.append(rule)
        return {'rules': rules, 'plan': {}}
    except Exception as e:
        return {'rules': [], 'plan': {}, '_parse_err': str(e)}


def run_eval(adapter_path: str, model_dir: str, qa_dataset_dir: str, out_dir: str, max_samples: int = None):
    """Load RuleVLM + iterate test.json + save preds_shard0.jsonl in SignVLM format.

    NOTE: chdir to RuleVLM_eval/ before loading because adapter_config.json has
    `base_model_name_or_path: "./qwen-vl-chat"` (relative path).
    """
    rulevlm_eval_dir = os.path.dirname(adapter_path).replace('/adapter_weights', '')
    # adapter_path looks like .../ckpts/RuleVLM_eval/adapter_weights/checkpoint-2020
    # so parent.parent is the RuleVLM_eval dir
    rulevlm_eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(adapter_path)))
    orig_cwd = os.getcwd()
    print(f'[eval] chdir to {rulevlm_eval_dir} (for peft relative base_model resolution)')
    os.chdir(rulevlm_eval_dir)
    # Resolve absolute paths now that cwd changed
    if not os.path.isabs(qa_dataset_dir):
        qa_dataset_dir = os.path.join(orig_cwd, qa_dataset_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(orig_cwd, out_dir)

    from transformers import AutoTokenizer
    from peft import AutoPeftModelForCausalLM

    test_data_path = os.path.join(qa_dataset_dir, 'test.json')
    uid2ts_path = os.path.join(qa_dataset_dir, 'uid2ts.json')

    with open(test_data_path) as f:
        test_data = json.load(f)
    with open(uid2ts_path) as f:
        uid2ts_dict = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f'[eval] {len(test_data)} samples')
    # Use relative paths so peft can resolve base_model from adapter_config.json
    rel_model = './qwen-vl-chat'
    rel_adapter = './adapter_weights/checkpoint-2020'
    print(f'[eval] loading tokenizer from {rel_model} (cwd: {os.getcwd()})')
    tokenizer = AutoTokenizer.from_pretrained(rel_model, trust_remote_code=True)
    print(f'[eval] loading PEFT model from {rel_adapter}')
    model = AutoPeftModelForCausalLM.from_pretrained(
        rel_adapter,
        device_map='cuda:0',
        trust_remote_code=True,
        bf16=True,
    ).eval()
    model.transformer.uid2ts_dict = uid2ts_dict
    # Inject correct mapdr_dir (config default 'path/to/mapdr' is placeholder)
    mapdr_dir_for_mee = os.environ.get('MAPDR_DIR_FOR_MEE', str(Path(__file__).resolve().parent.parent / 'data' / 'MapDR'))
    model.transformer.mapdr_dir = mapdr_dir_for_mee
    print(f'[eval] model loaded; injected uid2ts_dict ({len(uid2ts_dict)} entries) + mapdr_dir={mapdr_dir_for_mee}')

    os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(out_dir, 'preds_shard0.jsonl')
    fout = open(preds_path, 'w')

    n_total = 0
    n_parse_fail = 0
    t0 = time.time()

    for i, sample in enumerate(test_data):
        prompt = sample['conversations'][0]['value']
        gt_text = sample['conversations'][1]['value']
        # extract uid from prompt (format: `<|extra_0|>{uid}<|extra_1|>`)
        try:
            uid = prompt.split('<|extra_0|>')[1].split('<|extra_1|>')[0]
        except Exception:
            uid = sample.get('id', f'unknown_{i}')

        t_s = time.time()
        try:
            response, _ = model.chat(
                tokenizer, query=prompt, history=None, max_window_size=2048
            )
        except Exception as e:
            response = f'<<chat_error: {e}>>'
            n_parse_fail += 1

        # Parse RuleVLM output → SignVLM schema
        pred_signvlm = parse_rulevlm_response_to_signvlm_schema(response)
        pred_text = json.dumps(pred_signvlm, ensure_ascii=False)
        if pred_signvlm.get('_parse_err'):
            n_parse_fail += 1

        # GT is in RuleVLM dict format; also convert for v3 fairness
        gt_signvlm = parse_rulevlm_response_to_signvlm_schema(gt_text)
        gt_text_norm = json.dumps(gt_signvlm, ensure_ascii=False)

        elapsed = time.time() - t_s
        rec = {
            'scene_id': uid,
            'rep_ts': str(uid2ts_dict.get(uid, '')),
            'gt_text': gt_text_norm,
            'pred_text': pred_text,
            'raw_response': response[:2000],
            'new_tokens': len(response.split()),
            'elapsed_s': round(elapsed, 3),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        n_total += 1

        if i % 25 == 0:
            elapsed_total = time.time() - t0
            rate = n_total / max(elapsed_total, 1)
            eta = (len(test_data) - n_total) / max(rate, 1e-3) / 60
            print(f'[eval] {n_total}/{len(test_data)}  rate={rate:.2f}/s  parse_fail={n_parse_fail}  ETA={eta:.1f}min')

    fout.close()
    print(f'[eval] DONE  total={n_total}  parse_fail={n_parse_fail}  elapsed={(time.time()-t0)/60:.1f}min')
    print(f'[eval] preds written to {preds_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapdr-dir', required=True, help='Per-scene dirs root (data.json + label.json + img/)')
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--qa-dataset-dir', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--gen-data-only', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()

    if not args.eval_only:
        gen_qa_data(args.mapdr_dir, args.split_file, args.qa_dataset_dir, test_only=True)
        if args.gen_data_only:
            return

    run_eval(args.adapter, args.model_dir, args.qa_dataset_dir, args.out_dir, args.max_samples)


if __name__ == '__main__':
    main()
