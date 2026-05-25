"""Small planner: structured-input → 30-frame ego trajectory.

Architecture: per-modality MLP encoders → concat → trunk MLP → output.
~150K params, intentionally small to avoid overfit on 5613 train scenes.

Input feature_dict (from planner_extract_data.py):
- ego_history: (B, 8, 4)   x, y, yaw, v
- target_lane: (B, 20, 2)
- scene_lanes: (B, 6, 20, 2)
- scene_mask:  (B, 6)
- rule:        (B, 10)

Output: trajectory (B, 30, 2) — predicted ego (x, y)
"""
import torch
import torch.nn as nn


class PlannerMLP(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        # Per-modality encoders
        self.enc_ego = nn.Sequential(
            nn.Linear(8 * 4, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
        )
        self.enc_target_lane = nn.Sequential(
            nn.Linear(20 * 2, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
        )
        # Per-scene-lane encoder (shared across 6 lanes)
        self.enc_scene_lane = nn.Sequential(
            nn.Linear(20 * 2, 64), nn.GELU(),
            nn.Linear(64, 64),
        )
        self.enc_rule = nn.Sequential(
            nn.Linear(10, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 32),
        )
        # Trunk: 64 + 64 + 64 + 32 = 224 → hidden → ... → 30×2
        self.trunk = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 32, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 20 * 2),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        B = batch['ego_history'].shape[0]
        # Encode each modality
        h_ego = self.enc_ego(batch['ego_history'].reshape(B, -1))      # (B, 64)
        h_tgt = self.enc_target_lane(batch['target_lane'].reshape(B, -1))  # (B, 64)
        # Scene lanes: encode each, then mean-pool over valid lanes
        scene = batch['scene_lanes']  # (B, 6, 20, 2)
        mask = batch['scene_mask']    # (B, 6)
        scene_flat = scene.reshape(B * 6, 20 * 2)
        h_scene_each = self.enc_scene_lane(scene_flat).reshape(B, 6, 64)
        mask_exp = mask.unsqueeze(-1)  # (B, 6, 1)
        h_scene = (h_scene_each * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 64)
        h_rule = self.enc_rule(batch['rule'])  # (B, 32)
        # Concat + trunk
        h = torch.cat([h_ego, h_tgt, h_scene, h_rule], dim=-1)  # (B, 224)
        out = self.trunk(h)  # (B, 60)
        return out.reshape(B, 20, 2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = PlannerMLP()
    print(f'PlannerMLP params: {count_params(model):,}')
    # Smoke test
    batch = {
        'ego_history': torch.randn(2, 8, 4),
        'target_lane': torch.randn(2, 20, 2),
        'scene_lanes': torch.randn(2, 6, 20, 2),
        'scene_mask': torch.ones(2, 6),
        'rule': torch.randn(2, 10),
    }
    out = model(batch)
    print(f'output shape: {out.shape}')
    print(f'output stats: mean={out.mean():.3f} std={out.std():.3f}')
