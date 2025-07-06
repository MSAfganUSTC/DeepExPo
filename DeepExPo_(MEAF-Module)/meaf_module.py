"""
Implementation of the Multi‑Embedding Attention Fusion (MEAF) module and its
integration into a dual‑branch Stable‑Diffusion pipeline.

The code contains:

* **MEAF** – parallel attention streams that fuse identity & semantic
  embeddings with intermediate U‑Net features.
* **ZeroConv2d** – a 1×1 convolution whose weights & bias are initialised to
  zero, acting as a learnable yet safe gate between frozen and trainable
  branches.
* **DualBranchUNet** – wrapper that couples a **frozen** base U‑Net with a
  **trainable** copy where the original attention blocks are re‑configured to
  use MEAF; the two branches are joint together via two ZeroConvs.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.control_net import ControlNet
from nets.unet import UNetCondition2D, ControlUNet
from utils.canny import AddCannyImage
from utils.plotter import side_by_side_plot

from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup


class ZeroConv2d(nn.Conv2d):
    """1×1 convolution initialised to **zero** (weights & bias)."""

    def __init__(self, channels: int):
        super().__init__(channels, channels, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class MEAF(nn.Module):
    """Multi‑Embedding Attention Fusion (Eq. 1).

    Parameters
    ----------
    d_model : int
        Channel dimension (`d` in the paper).  Must match the #channels of the
        intermediate U‑Net feature map *and* the embedding size.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.scale = 1.0 / math.sqrt(d_model)

    @staticmethod
    def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """Scaled dot‑product attention with singleton (1‑token) K/V.

        Shapes
        -------
        q : (B, L, d)
        k : (B, 1, d)
        v : (B, 1, d)
        returns : (B, L, d)
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, L, 1)
        weights = torch.softmax(scores, dim=1)                 # (B, L, 1)
        return weights * v                                     # broadcast (B, L, d)

    def forward(
        self,
        feat: torch.Tensor,   # (B, C, H, W) – intermediate U‑Net features (Q)
        eid: torch.Tensor,    # (B, C)       – identity embedding  (K=V)
        esem: torch.Tensor,   # (B, C)       – semantic embedding (K=V)
    ) -> torch.Tensor:
        B, C, H, W = feat.shape
        assert C == self.d, "Channel mismatch between features and embeddings"

        # flatten spatial dims (B, HW, C) – acts as the query Q (F_{U‑Net}).
        q = feat.flatten(2).transpose(1, 2)

        # Prepare K/V tensors with singleton sequence length of 1.
        k_id = eid.unsqueeze(1)   # (B,1,C)
        v_id = eid.unsqueeze(1)
        k_se = esem.unsqueeze(1)
        v_se = esem.unsqueeze(1)

        # Parallel attention streams.
        out_id = self._attend(q, k_id, v_id, self.scale)
        out_se = self._attend(q, k_se, v_se, self.scale)

        # Fuse by element‑wise addition (Eq. 1).
        out = out_id + out_se        # (B, HW, C)

        # Restore spatial layout → (B, C, H, W).
        return out.transpose(1, 2).view(B, C, H, W)


class DualBranchUNet(nn.Module):
    """Couples a **frozen** base U‑Net with a **trainable** MEAF‑augmented copy."""

    def __init__(
        self,
        base_unet: nn.Module,
        trainable_unet: nn.Module,
        d_model: int,
        extract_feat: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        base_unet      : pretrained Stable‑Diffusion U‑Net (frozen)
        trainable_unet : copy of the U‑Net whose attention blocks have been
                         re‑configured to accept MEAF fusion.
        d_model        : channel dimension of the MEAF / U‑Net mid‑features.
        extract_feat   : if True, assumes `base_unet` returns both (y, feat).
                         Otherwise we take the *input* `x` as `F_{U‑Net}`.  The
                         latter is a simplification useful when the feature
                         extractor is wired inside `trainable_unet`.
        """
        super().__init__()

        # Freeze base branch
        self.base_unet = base_unet.eval()
        for p in self.base_unet.parameters():
            p.requires_grad = False

        # Trainable branch
        self.meaf = MEAF(d_model)
        self.zero1 = ZeroConv2d(d_model)
        self.trainable_unet = trainable_unet  # parameters are trainable by default
        self.zero2 = ZeroConv2d(d_model)

        self.extract_feat = extract_feat

    @torch.no_grad()
    def _base_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.extract_feat:
            y_base, feat = self.base_unet(x)
        else:
            y_base = self.base_unet(x)
            # use input as proxy for feature map (simplest integration)
            feat = x
        return y_base, feat

    def forward(
        self,
        x: torch.Tensor,        # noisy latent z_t
        eid: torch.Tensor,      # identity embedding
        esem: torch.Tensor,     # semantic embedding
    ) -> torch.Tensor:
        # 1) Frozen branch
        y_base, feat = self._base_forward(x)

        # 2) Trainable branch gated by ZeroConv‑1 and ZeroConv‑2
        fused = self.meaf(feat, eid, esem)           # E_fus in Eq. 3
        y_train = self.trainable_unet(x + self.zero1(fused))
        y_out = y_base + self.zero2(y_train)

        return y_out


class SDNoisePredLoss(nn.Module):
    """Stable‑Diffusion noise‑prediction loss (Eq. 4)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        eps_pred: torch.Tensor,  # ε_θ(z_t, …)
        eps_gt: torch.Tensor,    # true noise ε
    ) -> torch.Tensor:
        return self.mse(eps_pred, eps_gt)


class PerceptualLoss(nn.Module):
    """VGG‑19 perceptual loss used for identity preservation (Eq. 5)."""

    def __init__(self, vgg_model: nn.Module, layers: List[int]):
        super().__init__()
        self.vgg = vgg_model.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layers = layers
        self.criterion = nn.MSELoss()

    def _get_feats(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats, cur = [], x
        for i, layer in enumerate(self.vgg.features):
            cur = layer(cur)
            if i in self.layers:
                feats.append(cur)
            if len(feats) == len(self.layers):
                break
        return feats

    def forward(self, gen: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        feats_gen = self._get_feats(gen)
        feats_ref = self._get_feats(ref)
        loss = 0.0
        for fg, fr in zip(feats_gen, feats_ref):
            loss += self.criterion(fg, fr)
        return loss


class TotalLoss(nn.Module):
    """Weighted sum of SD noise‑prediction, perceptual, and MSE losses (Eq. 6)."""

    def __init__(
        self,
        vgg_body: nn.Module,
        percep_layers: List[int],
        lambda_percep: float = 1.0,
        lambda_mse: float = 1.0,
    ) -> None:
        super().__init__()
        self.sd_loss = SDNoisePredLoss()
        self.percep_loss = PerceptualLoss(vgg_body, percep_layers)
        self.mse_loss = nn.MSELoss()
        self.w_percep = lambda_percep
        self.w_mse = lambda_mse

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_gt: torch.Tensor,
        img_gen: torch.Tensor,
        img_id: torch.Tensor,
        img_ref_expr: torch.Tensor,
    ) -> torch.Tensor:
        l_sd = self.sd_loss(eps_pred, eps_gt)
        l_percep = self.percep_loss(img_gen, img_id)
        l_mse = self.mse_loss(img_gen, img_ref_expr)
        return l_sd + self.w_percep * l_percep + self.w_mse * l_mse

