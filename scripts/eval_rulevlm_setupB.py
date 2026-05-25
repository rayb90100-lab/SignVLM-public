"""RuleVLM Setup B eval: RuleVLM model × SignVLM panel-aware visual_prompt with conflict perturb.

Purpose: close reviewer attack "RuleVLM collapses too when given our conflict setup".
- cropped_sign: from SignVLM dataset (matches what SignVLM sees)
- visual_prompt: from SignVLM dataset with perturb_mode='conflict', seed=42 (panel-aware, single-field perturb)
- prompt format: RuleVLM's native (<|extra_*|>uid<|extra_*|>) — keep so RuleVLM model isn't OOD on prompt
- conflict_meta: SignVLM's single-field perturb (matches eval_wave7_7B_C03_seed42_panel_conflict for apples-to-apples)

Run under rulevlm conda env (transformers==4.42.3 pinned).
"""

import os
import sys
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RULEVLM_REPO = os.environ.get('RULEVLM_REPO', str(REPO.parent / 'RuleVLM_ref' / 'RuleVLM'))
SIGNVLM_SRC = str(REPO / 'src')


def build_qa_data(qa_dir: str):
    """Iterate SignVLM Test split with conflict perturb, save visual_prompt + cropped_sign,
    build RuleVLM-format test.json + uid2ts.json."""
    # SignVLM src must come FIRST — both repos define a `dataset` module; we need SignVLM's.
    # Build qa data uses SignVLM dataset; gen_data import is not needed here (Setup B doesn't
    # call RuleVLM's gen_data, only its model.chat in run_eval which happens in a different env).
    if SIGNVLM_SRC not in sys.path:
        sys.path.insert(0, SIGNVLM_SRC)

    # Force-clear any stale `dataset` import from RULEVLM_REPO
    for mod in list(sys.modules.keys()):
        if mod == 'dataset' or mod.startswith('dataset.'):
            del sys.modules[mod]

    from dataset import MapDRDataset  # noqa: E402

    vp_dir = os.path.join(qa_dir, 'visual_prompt_signvlm')
    cs_dir = os.path.join(qa_dir, 'cropped_signs_signvlm')
    os.makedirs(vp_dir, exist_ok=True)
    os.makedirs(cs_dir, exist_ok=True)

    ds = MapDRDataset(
        root=str(REPO / 'data' / 'MapDR'),
        split='Test',
        shuffle_idx=False,
        perturb_mode='conflict',
        perturb_prob=1.0,
        n_conflict_fields=1,
        seed=42,
    )

    qa_list = []
    uid2ts = {}

    for i in range(len(ds)):
        sample = ds[i]
        meta = sample['meta']
        uid = meta['scene_id']
        rep_ts = str(meta.get('rep_ts', ''))

        # Save images
        vp_path = os.path.join(vp_dir, f'{uid}.jpg')
        cs_path = os.path.join(cs_dir, f'{uid}.jpg')
        sample['visual_prompt'].save(vp_path, quality=85)
        sample['cropped_sign'].save(cs_path, quality=85)

        uid2ts[uid] = rep_ts

        # Build RuleVLM-format prompt (same as their gen_data.py line 382)
        prompt = (
            f'Picture 1: <img>{cs_path}</img>\n '
            f'Picture 2: <img>{vp_path}</img>\n '
            f'list1: <|extra_0|>{uid}<|extra_1|>, 使用(index: token)的形式表达;\n  '
            f'<|extra_2|>_<|extra_3|> \n '
            f'根据两张图像,预测 Picture1 中标牌所表达的若干条交通规则并且以 dict 输出, '
            f'Picture2 是其所在场景的前视图。每条规则中 centerline 的 value 应该是 list1 中对应的 index '
        )
        # Use SignVLM gt_text as answer (already in dict format with rules + plan)
        qa_list.append({
            'id': f'identity_{i}',
            'conversations': [
                {'from': 'user', 'value': prompt},
                {'from': 'assistant', 'value': sample['gt_text']},
            ],
            'conflict_meta': meta.get('conflict_meta'),
        })

        if i % 100 == 0:
            print(f'[gen] {i}/{len(ds)}')

    with open(os.path.join(qa_dir, 'test.json'), 'w') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    with open(os.path.join(qa_dir, 'uid2ts.json'), 'w') as f:
        json.dump(uid2ts, f)
    print(f'[gen] wrote {qa_dir}/test.json ({len(qa_list)} samples)')


def parse_rulevlm_response_to_signvlm_schema(response: str) -> dict:
    """Same conversion as eval_rulevlm.py."""
    try:
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
        for _, entry in rv_dict.items():
            if isinstance(entry, dict) and 'attr_info' in entry:
                rules.append({'attr_info': entry['attr_info'], 'centerline': entry.get('centerline', [])})
        return {'rules': rules, 'plan': {}}
    except Exception as e:
        return {'rules': [], 'plan': {}, '_parse_err': str(e)}


def run_eval(qa_dir: str, out_dir: str):
    """Load RuleVLM + iterate test.json + save preds_shard0.jsonl."""
    rulevlm_eval_dir = REPO / 'ckpts' / 'RuleVLM_eval'
    os.chdir(rulevlm_eval_dir)

    # CRITICAL: build_qa_data loaded SignVLM's `dataset.py` (single module),
    # but RuleVLM's modeling_qwen.py needs `dataset` to be the RuleVLM_ref package
    # (with submodule gen_data). Clear stale dataset import + ensure RULEVLM_REPO
    # is on sys.path first.
    for mod in list(sys.modules.keys()):
        if mod == 'dataset' or mod.startswith('dataset.'):
            del sys.modules[mod]
    # Remove SIGNVLM_SRC from sys.path so its dataset.py doesn't shadow RuleVLM's
    if SIGNVLM_SRC in sys.path:
        sys.path.remove(SIGNVLM_SRC)
    if RULEVLM_REPO not in sys.path:
        sys.path.insert(0, RULEVLM_REPO)
    init_file = Path(RULEVLM_REPO) / 'dataset' / '__init__.py'
    if not init_file.exists():
        init_file.touch()

    import torch  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from peft import AutoPeftModelForCausalLM  # noqa: E402

    with open(os.path.join(qa_dir, 'test.json')) as f:
        test_data = json.load(f)
    with open(os.path.join(qa_dir, 'uid2ts.json')) as f:
        uid2ts_dict = json.load(f)

    print(f'[eval] {len(test_data)} samples')
    tokenizer = AutoTokenizer.from_pretrained('./qwen-vl-chat', trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        './adapter_weights/checkpoint-2020',
        device_map='cuda:0',
        trust_remote_code=True,
        bf16=True,
    ).eval()
    model.transformer.uid2ts_dict = uid2ts_dict
    model.transformer.mapdr_dir = str(REPO / 'data' / 'MapDR')
    print(f'[eval] model loaded')

    os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(out_dir, 'preds_shard0.jsonl')
    fout = open(preds_path, 'w')
    t0 = time.time()
    parse_fail = 0

    for i, sample in enumerate(test_data):
        prompt = sample['conversations'][0]['value']
        gt_text = sample['conversations'][1]['value']
        try:
            uid = prompt.split('<|extra_0|>')[1].split('<|extra_1|>')[0]
        except Exception:
            uid = f'unknown_{i}'

        t_s = time.time()
        try:
            response, _ = model.chat(tokenizer, query=prompt, history=None, max_window_size=2048)
        except Exception as e:
            response = f'<<chat_error: {e}>>'
            parse_fail += 1

        pred_signvlm = parse_rulevlm_response_to_signvlm_schema(response)
        if pred_signvlm.get('_parse_err'):
            parse_fail += 1

        rec = {
            'scene_id': uid,
            'rep_ts': uid2ts_dict.get(uid, ''),
            'gt_text': gt_text,
            'pred_text': json.dumps(pred_signvlm, ensure_ascii=False),
            'raw_response': response[:2000],
            'conflict_meta': sample.get('conflict_meta'),
            'elapsed_s': round(time.time() - t_s, 3),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()

        if i % 25 == 0:
            r = (i + 1) / max(time.time() - t0, 1)
            eta = (len(test_data) - i - 1) / max(r, 1e-3) / 60
            print(f'[eval] {i+1}/{len(test_data)} rate={r:.2f}/s parse_fail={parse_fail} ETA={eta:.1f}min')

    fout.close()
    print(f'[eval] DONE {(time.time()-t0)/60:.1f}min, parse_fail={parse_fail}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa-dir', default='/tmp/rulevlm_setupB_qa')
    ap.add_argument('--out-dir', default=str(REPO / 'runs/rulevlm/eval_setupB_conflict'))
    ap.add_argument('--gen-only', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()

    if not args.eval_only:
        build_qa_data(args.qa_dir)
        if args.gen_only:
            return
    run_eval(args.qa_dir, args.out_dir)


if __name__ == '__main__':
    main()
