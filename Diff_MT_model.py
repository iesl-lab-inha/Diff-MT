"""Differential multimodal transformer for biosignal classification."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    aligned_steps: int = 64
    heads: int = 4
    window: int = 16
    hop: int = 8
    strides: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    fusion_weight: float = 1.0


class ModalityAlignment(nn.Module):
    """Project modality channels and align the temporal dimension."""
    def __init__(self, channels: int, cfg: ModelConfig):
        super().__init__()
        self.conv = nn.Conv1d(channels, cfg.d_model, kernel_size=3, padding=1)
        self.steps = cfg.aligned_steps

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError('Expected [batch, channels, time] input')
        x = self.conv(x)
        x = F.adaptive_avg_pool1d(x, self.steps)
        return x.transpose(1, 2)


class DMFE(nn.Module):
    """Overlapping-window, hierarchical temporal differencing and fusion."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.window, self.hop, self.strides = cfg.window, cfg.hop, cfg.strides
        if not 0 < self.hop <= self.window <= cfg.aligned_steps:
            raise ValueError('Require 0 < hop <= window <= aligned_steps')
        if not self.strides or any(s <= 0 or s >= self.window for s in self.strides):
            raise ValueError('Each stride must lie in [1, window-1]')
        self.fuse = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model),
                                  nn.LayerNorm(cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t, d = x.shape
        w, hop = self.window, self.hop
        pad = (-(t - w)) % hop
        xp = F.pad(x.transpose(1, 2), (0, pad), mode='replicate')
        windows = xp.unfold(-1, w, hop)  # B,D,n,W
        delta = torch.zeros_like(windows)
        for stride in self.strides:
            # Compare each location with a pooled context at the current scale.
            pooled = F.avg_pool1d(windows.reshape(-1, 1, w), stride,
                                  stride=stride, ceil_mode=True)
            pooled = F.interpolate(pooled, size=w, mode='nearest').reshape_as(windows)
            diff = windows - pooled
            lag = F.pad(windows[..., stride:] - windows[..., :-stride], (stride, 0))
            delta = delta + diff + lag
        delta = delta / len(self.strides)
        # Overlap-add with normalization to avoid over-weighting central samples.
        cols = delta.permute(0, 1, 3, 2).reshape(b, d*w, -1)
        size = t + pad
        summed = F.fold(cols, (1, size), (1, w), stride=(1, hop)).squeeze(2)
        count = F.fold(x.new_ones(1, w, windows.shape[2]), (1, size),
                       (1, w), stride=(1, hop)).squeeze(2)
        diff = (summed / count.clamp_min(1))[..., :t].transpose(1, 2)
        return self.fuse(torch.cat((diff, x), dim=-1)), diff


def positional_encoding(length: int, dim: int, device, dtype) -> Tensor:
    p = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    rate = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                     * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(p * rate)
    pe[:, 1::2] = torch.cos(p * rate[:dim//2])
    return pe.to(dtype)


class GCE(nn.Module):
    """Each modality queries the global sequence of modality features."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.attention = nn.MultiheadAttention(cfg.d_model, cfg.heads,
                                               dropout=cfg.dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, streams: list[Tensor]) -> Tensor:
        positioned = [self.pre_norm(x + positional_encoding(x.size(1), x.size(2),
                        x.device, x.dtype)) for x in streams]
        context = torch.cat(positioned, dim=1)
        outputs = []
        for x, query in zip(streams, positioned):
            z, _ = self.attention(query, context, context, need_weights=False)
            outputs.append(self.out_norm(x + self.dropout(z)))
        return torch.cat(outputs, dim=1)


class SelfAttentionPool(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        z, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + z).mean(dim=1)


class DWF(nn.Module):
    """Dual self-attention pooling and weighted differential/global fusion."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.differential = SelfAttentionPool(cfg)
        self.global_pool = SelfAttentionPool(cfg)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.fusion_weight)))

    def forward(self, differential: Tensor | None, global_features: Tensor) -> Tensor:
        global_pooled = self.alpha * self.global_pool(global_features)
        if differential is None:
            return global_pooled
        return self.differential(differential) + global_pooled


class DiffMT(nn.Module):
    def __init__(self, channels: dict[str, int], num_classes: int,
                 cfg: ModelConfig = ModelConfig(), variant: str = 'full'):
        super().__init__()
        if variant not in {'full', 'no_dmfe', 'no_gce', 'no_dwf', 'early'}:
            raise ValueError(f'Unknown variant: {variant}')
        if not channels or num_classes < 2:
            raise ValueError('At least one modality and two classes required')
        self.modalities, self.variant = tuple(channels), variant
        self.alignment = nn.ModuleDict({m: ModalityAlignment(c, cfg) for m,c in channels.items()})
        self.dmfe = (nn.ModuleDict({m: DMFE(cfg) for m in channels})
                     if variant not in {'no_dmfe', 'early'} else None)
        self.gce = GCE(cfg) if variant not in {'no_gce', 'early'} else None
        self.dwf = DWF(cfg) if variant not in {'no_dwf', 'early'} else None
        self.early = nn.Linear(len(channels)*cfg.d_model, cfg.d_model) if variant == 'early' else None
        self.hidden = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout))
        self.classifier = nn.Linear(cfg.d_model, num_classes)

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        aligned = [self.alignment[m](inputs[m]) for m in self.modalities]
        if self.early is not None:
            fused = self.early(torch.cat(aligned, dim=-1)).mean(dim=1)
        else:
            if self.dmfe is not None:
                pairs = [self.dmfe[m](x) for m, x in zip(self.modalities, aligned)]
                streams = [p[0] for p in pairs]
                differences = torch.cat([p[1] for p in pairs], dim=1)
            else:
                streams, differences = aligned, None
            global_features = self.gce(streams) if self.gce is not None else torch.cat(streams, dim=1)
            fused = self.dwf(differences, global_features) if self.dwf is not None else global_features.mean(dim=1)
        return self.classifier(fused + self.hidden(fused))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Tensor | None = None):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.gamma = gamma
        self.register_buffer('alpha', alpha)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        logp = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
        loss = -(1 - logp.exp()).pow(self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha[labels]
        return loss.mean()
