# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""AFM 特征增强模块及其必要辅助组件。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ("AFM",)


class _ConvBNAct(nn.Module):
    """卷积、批归一化和激活的组合。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _ECA(nn.Module):
    """ECA-Net 的无降维通道交互。"""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log2(max(channels, 1)) + b) / gamma))
        kernel_size = max(t if t % 2 else t + 1, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return y


class AFM(nn.Module):
    """AFM：以正向增强为主、受限背景抑制为辅的特征增强模块。

    正向路径包含通道增强与空间增强。背景路径仅在通道和空间分支
    同时给出低响应时生效，并通过有上限的标量限制抑制强度。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        temperature: float = 1.0,
        suppression_max: float = 0.1,
    ) -> None:
        super().__init__()
        del reduction
        if spatial_kernel < 1 or spatial_kernel % 2 == 0:
            raise ValueError(f"spatial_kernel 必须是正奇数，当前为 {spatial_kernel}")
        if temperature <= 0:
            raise ValueError(f"temperature 必须大于 0，当前为 {temperature}")
        if not 0.0 < suppression_max <= 1.0:
            raise ValueError(f"suppression_max 必须位于 (0, 1]，当前为 {suppression_max}")

        self.depthwise = _ConvBNAct(c1, c1, k=3, g=c1, act=nn.ReLU(inplace=True))
        self.pointwise = _ConvBNAct(c1, c2, k=1, act=nn.ReLU(inplace=True))
        self.eca = _ECA(c2)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))

        # 通道与空间分支共享池化参数，融合正向增强特征。
        self.pool_logits = nn.Parameter(torch.ones(2))
        self.spatial_cbs = _ConvBNAct(1, 1, k=spatial_kernel)
        self.shortcut = _ConvBNAct(c1, c2, k=1, act=False) if c1 != c2 else nn.Identity()

        # 初始抑制强度为上限的一半，训练过程中始终限制在 [0, suppression_max]。
        self.suppression_max = float(suppression_max)
        self.suppression_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x_dsc = self.pointwise(self.depthwise(x))

        temperature = self.log_temperature.exp().clamp_min(1e-4)
        channel_weight = torch.sigmoid(self.eca(x_dsc) / temperature)
        x_ca = x_dsc * channel_weight

        avg_out = x_ca.mean(dim=1, keepdim=True)
        max_out = x_ca.amax(dim=1, keepdim=True)
        alpha, beta = self.pool_logits.unbind()
        spatial_weight = torch.sigmoid(self.spatial_cbs(alpha * avg_out + beta * max_out))
        x_sa = x_ca * spatial_weight

        omega = self.pool_logits.softmax(dim=0)
        positive_enhancement = omega[0] * x_ca + omega[1] * x_sa

        # 仅当通道和空间证据均较弱时抑制背景，且不使用可翻转符号的逐通道缩放。
        background_gate = (1.0 - channel_weight) * (1.0 - spatial_weight)
        suppression_scale = self.suppression_max * self.suppression_logit.sigmoid()
        background_suppression = suppression_scale * x_dsc * background_gate
        return self.shortcut(x_orig) + positive_enhancement - background_suppression
