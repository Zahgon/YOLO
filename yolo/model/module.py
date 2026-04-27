from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from yolo.utils.logger import logger
from yolo.utils.module_utils import auto_pad, create_activation_function, round_up


# ----------- Basic Class ----------- #
class Conv(nn.Module):
    """A basic convolutional block that includes convolution, batch normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        self.act = create_activation_function(activation)

    def forward(self, x: Tensor) -> Tensor:
        pass


class Pool(nn.Module):
    """A generic pooling block supporting 'max' and 'avg' pooling methods."""

    def __init__(self, method: str = "max", kernel_size: _size_2_t = 2, **kwargs):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        pool_classes = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        self.pool = pool_classes[method.lower()](kernel_size=kernel_size, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        pass


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        pass


# ----------- Detection Class ----------- #
class Detection(nn.Module):
    """A single YOLO Detection head for detection models"""

    def __init__(self, in_channels: Tuple[int], num_classes: int, *, reg_max: int = 16, use_group: bool = True):
        super().__init__()

        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max

        first_neck, in_channels = in_channels
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        class_neck = max(first_neck, min(num_classes * 2, 128))

        self.anchor_conv = nn.Sequential(
            Conv(in_channels, anchor_neck, 3),
            Conv(anchor_neck, anchor_neck, 3, groups=groups),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
        )
        self.class_conv = nn.Sequential(
            Conv(in_channels, class_neck, 3), Conv(class_neck, class_neck, 3), nn.Conv2d(class_neck, num_classes, 1)
        )

        self.anc2vec = Anchor2Vec(reg_max=reg_max)

        self.anchor_conv[-1].bias.data.fill_(1.0)
        self.class_conv[-1].bias.data.fill_(-10)  # TODO: math.log(5 * 4 ** idx / 80 ** 3)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        pass


class IDetection(nn.Module):
    def __init__(self, in_channels: Tuple[int], num_classes: int, *args, anchor_num: int = 3, **kwargs):
        super().__init__()

        if isinstance(in_channels, tuple):
            in_channels = in_channels[1]

        out_channel = num_classes + 5
        out_channels = out_channel * anchor_num
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.implicit_a = ImplicitA(in_channels)
        self.implicit_m = ImplicitM(out_channels)

    def forward(self, x):
        pass


class MultiheadDetection(nn.Module):
    """Mutlihead Detection module for Dual detect or Triple detect"""

    def __init__(self, in_channels: List[int], num_classes: int, **head_kwargs):
        super().__init__()
        DetectionHead = Detection

        if head_kwargs.pop("version", None) == "v7":
            DetectionHead = IDetection

        self.heads = nn.ModuleList(
            [DetectionHead((in_channels[0], in_channel), num_classes, **head_kwargs) for in_channel in in_channels]
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        pass


# ----------- Segmentation Class ----------- #
class Segmentation(nn.Module):
    def __init__(self, in_channels: Tuple[int], num_maskes: int):
        super().__init__()
        first_neck, in_channels = in_channels

        mask_neck = max(first_neck // 4, num_maskes)
        self.mask_conv = nn.Sequential(
            Conv(in_channels, mask_neck, 3), Conv(mask_neck, mask_neck, 3), nn.Conv2d(mask_neck, num_maskes, 1)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        pass


class MultiheadSegmentation(nn.Module):
    """Mutlihead Segmentation module for Dual segment or Triple segment"""

    def __init__(self, in_channels: List[int], num_classes: int, num_maskes: int, **head_kwargs):
        super().__init__()
        mask_channels, proto_channels = in_channels[:-1], in_channels[-1]

        self.detect = MultiheadDetection(mask_channels, num_classes, **head_kwargs)
        self.heads = nn.ModuleList(
            [Segmentation((in_channels[0], in_channel), num_maskes) for in_channel in mask_channels]
        )
        self.heads.append(Conv(proto_channels, num_maskes, 1))

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        pass


class Anchor2Vec(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anc2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anc2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)

    def forward(self, anchor_x: Tensor) -> Tensor:
        pass


# ----------- Classification Class ----------- #
class Classification(nn.Module):
    def __init__(self, in_channel: int, num_classes: int, *, neck_channels=1024, **head_args):
        super().__init__()
        self.conv = Conv(in_channel, neck_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(neck_channels, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        pass


# ----------- Backbone Class ----------- #
class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        *,
        activation: Optional[str] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.act = create_activation_function(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False, **kwargs)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        pass


class Bottleneck(nn.Module):
    """A bottleneck block with optional residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv(neck_channels, out_channels, kernel_size[1], **kwargs)
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            logger.warning(
                "Residual connection disabled: in_channels ({}) != out_channels ({})", in_channels, out_channels
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RepNCSP(nn.Module):
    """RepNCSP block with convolutions, split, and bottleneck processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()

        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size, **kwargs)

        self.bottleneck = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ELAN(nn.Module):
    """ELAN  structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = Conv(part_channels // 2, process_channels, 3, padding=1, **kwargs)
        self.conv3 = Conv(process_channels, process_channels, 3, padding=1, **kwargs)
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        csp_args: Dict[str, Any] = {},
        csp_neck_args: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class AConv(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv = Conv(in_channels, out_channels, **mid_layer)

    def forward(self, x: Tensor) -> Tensor:
        pass


class ADown(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv1 = Conv(half_in_channels, half_out_channels, **mid_layer)
        self.max_pool = Pool("max", **mid_layer)
        self.conv2 = Conv(half_in_channels, half_out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        pass


class CBLinear(nn.Module):
    """Convolutional block that outputs multiple feature maps split along the channel dimension."""

    def __init__(self, in_channels: int, out_channels: List[int], kernel_size: int = 1, **kwargs):
        super(CBLinear, self).__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, sum(out_channels), kernel_size, **kwargs)
        self.out_channels = list(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        pass


class SPPCSPConv(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels: int, out_channels: int, expand: float = 0.5, kernel_sizes: Tuple[int] = (5, 9, 13)):
        super().__init__()
        neck_channels = int(2 * out_channels * expand)
        self.pre_conv = nn.Sequential(
            Conv(in_channels, neck_channels, 1),
            Conv(neck_channels, neck_channels, 3),
            Conv(neck_channels, neck_channels, 1),
        )
        self.short_conv = Conv(in_channels, neck_channels, 1)
        self.pools = nn.ModuleList([Pool(kernel_size=kernel_size, stride=1) for kernel_size in kernel_sizes])
        self.post_conv = nn.Sequential(Conv(4 * neck_channels, neck_channels, 1), Conv(neck_channels, neck_channels, 3))
        self.merge_conv = Conv(2 * neck_channels, out_channels, 1)

    def forward(self, x):
        pass


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: Optional[int] = None):
        super(SPPELAN, self).__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv(in_channels, neck_channels, kernel_size=1)
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv(4 * neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        pass


class UpSample(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.UpSample = nn.Upsample(**kwargs)

    def forward(self, x):
        pass


class CBFuse(nn.Module):
    def __init__(self, index: List[int], mode: str = "nearest"):
        super().__init__()
        self.idx = index
        self.mode = mode

    def forward(self, x_list: List[torch.Tensor]) -> List[Tensor]:
        pass


class ImplicitA(nn.Module):
    """
    Implement YOLOR - implicit knowledge(Add), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 0.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ImplicitM(nn.Module):
    """
    Implement YOLOR - implicit knowledge(multiply), paper: https://arxiv.org/abs/2105.04206
    """

    def __init__(self, channel: int, mean: float = 1.0, std: float = 0.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std

        self.implicit = nn.Parameter(torch.empty(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DConv(nn.Module):
    def __init__(self, in_channels=512, alpha=0.8, atoms=512):
        super().__init__()
        self.alpha = alpha

        self.CG = Conv(in_channels, atoms, 1)
        self.GIE = Conv(atoms, atoms, 5, groups=atoms, activation=False)
        self.D = Conv(atoms, in_channels, 1, activation=False)

    def PONO(self, x):
        pass

    def forward(self, r):
        pass


class RepNCSPELAND(RepNCSPELAN):
    def __init__(self, *args, atoms: 512, rd_args={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.dconv = DConv(atoms=atoms, **rd_args)

    def forward(self, x):
        pass
