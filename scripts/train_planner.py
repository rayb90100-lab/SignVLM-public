"""Train planner on MapDR extracted dataset.

Usage:
    python scripts/train_planner.py [--epochs 100] [--batch 128] [--lr 3e-4]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from planner_model import PlannerMLP, count_params

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get('PLANNER_DATA_ROOT', str(REPO / 'data' / 'planner_data')))
CKPT_ROOT = Path(os.environ.get('PLANNER_CKPT_ROOT', str(REPO / 'runs' / 'planner')))


class PlannerDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.files = sorted(split_dir.glob('*.pt'))
        if not self.files:
            raise RuntimeError(f'no .pt files in {split_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)


def collate(samples: list[dict]) -> dict:
    keys = ['ego_history', 'target_lane', 'scene_lanes', 'scene_mask', 'rule', 'gt_trajectory']
    out = {k: torch.stack([s[k] for s in samples]) for k in keys}
    out['uids'] = [s['uid'] for s in samples]
    return out


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """pred, gt: (B, 30, 2). Return ADE, FDE per sample mean."""
    dist = torch.norm(pred - gt, dim=-1)  # (B, 30)
    ade = dist.mean(dim=-1)  # (B,)
    fde = dist[:, -1]        # (B,)
    return {'ade': ade.mean().item(), 'fde': fde.mean().item()}


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    ades, fdes, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            pred = model(batch)
            loss = ((pred - batch['gt_trajectory']) ** 2).mean()
            m = compute_metrics(pred, batch['gt_trajectory'])
            ades.append(m['ade']); fdes.append(m['fde']); losses.append(loss.item())
    return {'loss': np.mean(losses), 'ade': np.mean(ades), 'fde': np.mean(fdes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--out-dir', default=None,
                    help='Run dir; default = runs/planner/<timestamp>')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.out_dir) if args.out_dir else CKPT_ROOT / time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'[train] run_dir: {run_dir}')
    print(f'[train] device: {device}')

    train_ds = PlannerDataset(DATA_ROOT / 'Train')
    test_ds = PlannerDataset(DATA_ROOT / 'Test')
    print(f'[train] train: {len(train_ds)} scenes; test: {len(test_ds)} scenes')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True
    )

    model = PlannerMLP(hidden=256, dropout=0.1).to(device)
    print(f'[train] params: {count_params(model):,}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = {'epoch': [], 'train_loss': [], 'test_loss': [], 'test_ade': [], 'test_fde': [], 'lr': []}
    best_test_ade = float('inf')

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f'epoch {epoch+1}/{args.epochs}', leave=False)
        for batch in pbar:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            pred = model(batch)
            loss = ((pred - batch['gt_trajectory']) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        sched.step()
        train_loss = float(np.mean(epoch_losses))

        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
        if do_eval:
            ev = evaluate(model, test_loader, device)
            log['epoch'].append(epoch + 1)
            log['train_loss'].append(train_loss)
            log['test_loss'].append(ev['loss'])
            log['test_ade'].append(ev['ade'])
            log['test_fde'].append(ev['fde'])
            log['lr'].append(opt.param_groups[0]['lr'])
            elapsed = time.time() - t0
            print(f'[ep {epoch+1:3d}] train={train_loss:.4f}  test_loss={ev["loss"]:.4f}  '
                  f'test_ade={ev["ade"]:.3f}m  test_fde={ev["fde"]:.3f}m  '
                  f'lr={opt.param_groups[0]["lr"]:.2e}  elapsed={elapsed:.0f}s')
            if ev['ade'] < best_test_ade:
                best_test_ade = ev['ade']
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1,
                            'test_ade': ev['ade'], 'test_fde': ev['fde'],
                            'args': vars(args)},
                           run_dir / 'best.pt')
        with open(run_dir / 'log.json', 'w') as f:
            json.dump(log, f, indent=2)

    print(f'\n[train] DONE  best test_ade={best_test_ade:.3f}m  total={time.time()-t0:.0f}s')
    torch.save({'model': model.state_dict(), 'epoch': args.epochs,
                'args': vars(args)},
               run_dir / 'final.pt')
    print(f'[train] saved {run_dir}/final.pt')


if __name__ == '__main__':
    main()
