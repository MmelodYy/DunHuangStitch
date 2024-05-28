import torch
import copy
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F


class RepBlock(nn.Module):
    """
    MobileOne-style residual blocks, including residual joins and re-parameterization convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        rbr_conv_kernel_list: List[int] = [7, 3],
        use_bn_conv: bool = False,
        act_layer: nn.Module = nn.ReLU,
        skip_include_bn: bool = True,
    ) -> None:
        """Construct a Re-parameterization module.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for convolution.
        :param groups: Number of groups for convolution.
        :param inference_mode: Whether to use inference mode.
        :param rbr_conv_kernel_list: List of kernel sizes for re-parameterizable convolutions.
        :param use_bn_conv: Whether the bn is in front of conv, if false, conv is in front of bn
        :param act_layer: Activation layer.
        :param skip_include_bn: Whether to include bn in skip connection.
        """
        super(RepBlock, self).__init__()

        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rbr_conv_kernel_list = sorted(rbr_conv_kernel_list, reverse=True)
        self.num_conv_branches = len(self.rbr_conv_kernel_list)
        self.kernel_size = self.rbr_conv_kernel_list[0]
        self.use_bn_conv = use_bn_conv
        self.skip_include_bn = skip_include_bn

        self.activation = act_layer()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.kernel_size // 2,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            if out_channels == in_channels and stride == 1:
                if self.skip_include_bn:
                    # Use residual connections that include BN
                    self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)
                else:
                    # Use residual connections
                    self.rbr_skip = nn.Identity()
            else:
                # Use residual connections
                self.rbr_skip = None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for kernel_size in self.rbr_conv_kernel_list:
                if self.use_bn_conv:
                    rbr_conv.append(
                        self._bn_conv(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )
                else:
                    rbr_conv.append(
                        self._conv_bn(
                            in_chans=in_channels,
                            out_chans=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=kernel_size // 2,
                            groups=groups,
                        )
                    )

            self.rbr_conv = nn.ModuleList(rbr_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Other branches
        out = identity_out
        for ix in range(self.num_conv_branches):
            out = out + self.rbr_conv[ix](x)
        return self.activation(out)

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.rbr_conv[0].conv.in_channels,
            out_channels=self.rbr_conv[0].conv.out_channels,
            kernel_size=self.rbr_conv[0].conv.kernel_size,
            stride=self.rbr_conv[0].conv.stride,
            padding=self.rbr_conv[0].conv.padding,
            dilation=self.rbr_conv[0].conv.dilation,
            groups=self.rbr_conv[0].conv.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_skip_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            if self.use_bn_conv:
                _kernel, _bias = self._fuse_bn_conv_tensor(self.rbr_conv[ix])
            else:
                _kernel, _bias = self._fuse_conv_bn_tensor(self.rbr_conv[ix])
            # pad kernel
            if _kernel.shape[-1] < self.kernel_size:
                pad = (self.kernel_size - _kernel.shape[-1]) // 2
                _kernel = torch.nn.functional.pad(_kernel, [pad, pad, pad, pad])

            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_identity
        bias_final = bias_conv + bias_identity
        return kernel_final, bias_final

    def _fuse_skip_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param branch: skip branch, maybe include bn layer
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """

        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                dtype=self.rbr_conv[0].conv.weight.dtype,
                device=self.rbr_conv[0].conv.weight.device,
            )
            for i in range(self.in_channels):
                kernel_value[
                    i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                ] = 1
            self.id_tensor = kernel_value
        if isinstance(branch, nn.Identity):
            kernel = self.id_tensor
            return kernel, torch.zeros(
                (self.in_channels),
                dtype=self.rbr_conv[0].conv.weight.dtype,
                device=self.rbr_conv[0].conv.weight.device,
            )
        else:
            assert isinstance(
                branch, nn.BatchNorm2d
            ), "Make sure the module in skip is nn. BatchNorm2d"
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_conv_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """先bn,后conv

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = gamma / std
        t = torch.stack([t] * (kernel.shape[0] * kernel.shape[1] // t.shape[0]), dim=0).reshape(-1,
                                                                                                self.in_channels // self.groups,
                                                                                                1, 1)
        t_beta = torch.stack([beta] * (kernel.shape[0] * kernel.shape[1] // beta.shape[0]), dim=0).reshape(-1,
                                                                                                           self.in_channels // self.groups,
                                                                                                           1, 1)
        t_running_mean = torch.stack([running_mean] * (kernel.shape[0] * kernel.shape[1] // running_mean.shape[0]),
                                     dim=0).reshape(-1, self.in_channels // self.groups, 1, 1)
        return kernel * t, torch.sum(
            kernel
            * (
                t_beta - t_running_mean * t
            ),
            dim=(1, 2, 3),
        )

    def _fuse_conv_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """First conv, then bn

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """

        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ) -> nn.Sequential:
        """First conv, then bn

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=out_chans))
        return mod_list

    def _bn_conv(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ) -> nn.Sequential:
        """Add bn first, then conv"""
        mod_list = nn.Sequential()
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=in_chans))
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        )
        return mod_list


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttnTokenMixer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        num_heads: int,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        inference_mode: bool = False,
        use_CPE: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_chans)
        self.attn = Attention(
            in_chans,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
        )
        self.norm = nn.LayerNorm(in_chans)
        if use_CPE:
            self.CPE = RepBlock(
                in_channels=in_chans,
                out_channels=in_chans,
                rbr_conv_kernel_list=[3],
                stride=1,
                groups=in_chans,
                inference_mode=inference_mode,
                act_layer=nn.Identity,
                skip_include_bn=False,
            )
        else:
            self.CPE = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hp, Wp = x.shape
        x = self.CPE(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.attn(self.norm(x))
        x = x.permute(0, 2, 1).reshape(B, C, Hp, Wp).contiguous()
        return x


class FastViTBlock(nn.Module):
    def __init__(
        self,
        chans: int,
        num_heads: int,
        inference_mode: bool,
        act_layer: nn.Module,
        use_attn: bool,
        expand_ratio: int,
        use_CPE: bool = False,
    ) -> None:
        super().__init__()
        if use_attn:
            # AttnTokenMixer
            self.token_mixer = AttnTokenMixer(
                chans, num_heads, inference_mode=inference_mode, use_CPE=use_CPE
            )
        else:
            # RepMixer
            self.token_mixer = RepBlock(
                in_channels=chans,
                out_channels=chans,
                stride=1,
                groups=chans,
                rbr_conv_kernel_list=[3],
                use_bn_conv=True,
                act_layer=act_layer,
                inference_mode=inference_mode,
                skip_include_bn=False,
            )
        mid_chans = chans * expand_ratio
        # conv_ffn
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=7, padding=3, groups=chans, bias=False),
            nn.BatchNorm2d(chans),
            nn.Conv2d(chans, mid_chans, kernel_size=1, padding=0),
            act_layer(),
            nn.Conv2d(mid_chans, chans, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = x + self.conv_ffn(x)
        return x


class DwonStage(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        use_attn: bool,
        num_heads: int,
        num_blocks_per_stage: int,
        inference_mode: bool,
        act_layer: nn.Module,
        expand_ratio: int,
        use_patch_embed: bool,
    ) -> None:
        """
        Constructs a FastStage

        :param in_chans: Number of input channels.
        :param out_chans: Number of output channels.
        :param num_heads: Number of heads for attention If use_attn is True.
        :param use_attn: Whether to use attention.
        :param num_blocks_per_stage: Number of blocks per stage.
        :param inference_mode: Whether to use inference mode.
        :param act_layer: Activation layer.
        :param expand_ratio: Expansion ratio in conv_ffn.
        :param use_patch_embed: Whether to use patch embedding.
        """
        super().__init__()
        self.num_blocks_per_stage = num_blocks_per_stage
        if use_patch_embed:
            self.patch_embed = nn.Sequential(
                RepBlock(
                    in_channels=in_chans,
                    out_channels=in_chans,
                    rbr_conv_kernel_list=[7, 3],
                    stride=2,
                    groups=in_chans,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                ),
                RepBlock(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                ),
            )
        else:
            self.patch_embed = nn.Identity()

        # FastViTBlock, CPE is only used on the first block
        self.blocks = nn.Sequential(
            *[
                FastViTBlock(
                    chans=out_chans,
                    num_heads=num_heads,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_attn=use_attn,
                    expand_ratio=expand_ratio,
                    use_CPE=(i == 0),
                )
                for i in range(num_blocks_per_stage)
            ]
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x


class UpStage(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        use_attn: bool,
        num_heads: int,
        num_blocks_per_stage: int,
        inference_mode: bool,
        act_layer: nn.Module,
        expand_ratio: int,
        use_patch_embed: bool,
    ) -> None:
        """
        Constructs a FastStage

        :param in_chans: Number of input channels.
        :param out_chans: Number of output channels.
        :param num_heads: Number of heads for attention If use_attn is True.
        :param use_attn: Whether to use attention.
        :param num_blocks_per_stage: Number of blocks per stage.
        :param inference_mode: Whether to use inference mode.
        :param act_layer: Activation layer.
        :param expand_ratio: Expansion ratio in conv_ffn.
        :param use_patch_embed: Whether to use patch embedding.
        """
        super().__init__()
        self.num_blocks_per_stage = num_blocks_per_stage
        if use_patch_embed:
            self.patch_embed = nn.Sequential(
                RepBlock(
                    in_channels=in_chans,
                    out_channels=in_chans,
                    rbr_conv_kernel_list=[7, 3],
                    stride=1,
                    groups=in_chans,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                ),
                RepBlock(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    rbr_conv_kernel_list=[1],
                    stride=1,
                    groups=1,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                ),
                nn.Upsample(scale_factor=2)
            )
        else:
            self.patch_embed = nn.Identity()

        # FastViTBlock, CPE is only used on the first block
        self.blocks = nn.Sequential(
            *[
                FastViTBlock(
                    chans=out_chans,
                    num_heads=num_heads,
                    inference_mode=inference_mode,
                    act_layer=act_layer,
                    use_attn=use_attn,
                    expand_ratio=expand_ratio,
                    use_CPE=(i == 0),
                )
                for i in range(num_blocks_per_stage)
            ]
        )

    def forward(self, x_up1, skip):
        # print("x_up1:",x_up1.shape)
        x_up1 = self.patch_embed(x_up1)
        x_up1 = F.interpolate(x_up1, size=(skip.shape[2], skip.shape[3]))
        # print("x_up1:",x_up1.shape)
        # print("skip:",skip.shape)
        x = x_up1 + skip
        x = self.blocks(x)
        return x


class FastStitch(nn.Module):
    def __init__(
        self,
        inchannels: int = 3,
        inference_mode: bool = False,
        in_chans_list: Tuple[int] = (48, 48, 96, 192),
        out_chans_list: Tuple[int] = (48, 96, 192, 384),
        blocks_per_stage: Tuple[int] = (2, 2, 4, 2),
        expand_ratio: Tuple[int] = (4, 4, 4, 4),
        use_attn: Tuple[bool] = (False, False, False, False),
        use_patchEmb: Tuple[bool] = (False, True, True, True),
        act_layer: nn.Module = nn.ReLU,
    ) -> None:
        """
        Constructs a FastVit model

        :param num_classes: Number of classes for classification head.
        :param inference_mode: Whether to use inference mode.
        :param in_chans_list: List of input channels for each stage.
        :param out_chans_list: List of output channels for each stage.
        :param blocks_per_stage: List of number of blocks for each stage.
        :param expand_ratio: List of expansion ratios for each stage.
        :param use_attn: List of whether to use attention for each stage.
        :param use_patchEmb: List of whether to use patch embedding for each stage.
        :param act_layer: Activation layer.
        """

        super().__init__()

        self.stem = nn.Sequential(
            RepBlock(
                in_channels=inchannels,
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=2,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
                skip_include_bn=True,
            ),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=2,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
                skip_include_bn=True,
            ),
        )

        self.down = nn.Sequential(
            *(
                DwonStage(
                    in_chans=in_chans_list[i],
                    out_chans=out_chans_list[i],
                    num_blocks_per_stage=blocks_per_stage[i],
                    inference_mode=inference_mode,
                    use_attn=use_attn[i],
                    num_heads=8,
                    expand_ratio=expand_ratio[i],
                    act_layer=act_layer,
                    use_patch_embed=use_patchEmb[i],
                )
                for i in range(len(blocks_per_stage))
            )
        )

        self.up = nn.Sequential(
            *(
                UpStage(
                    in_chans=out_chans_list[i],
                    out_chans=in_chans_list[i],
                    num_blocks_per_stage=blocks_per_stage[i],
                    inference_mode=inference_mode,
                    use_attn=use_attn[i],
                    num_heads=8,
                    expand_ratio=expand_ratio[i],
                    act_layer=act_layer,
                    use_patch_embed=use_patchEmb[i],
                )
                for i in range(1, len(blocks_per_stage))
            )
        )[::-1]

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            nn.Upsample(scale_factor=2),
            RepBlock(
                in_channels=in_chans_list[0],
                out_channels=in_chans_list[0],
                rbr_conv_kernel_list=[3, 1],
                stride=1,
                groups=1,
                inference_mode=inference_mode,
                act_layer=act_layer,
            ),
            nn.Conv2d(in_chans_list[0], 1, 1),
            nn.Sigmoid()
        )

    def forward(self, or_img1, or_img2, mask1, mask2):
        # share weight stem
        x1 = self.stem(or_img1)
        x2 = self.stem(or_img2)
        x = x2 - x1

        # Down
        f_list = []
        f_down = x
        for i in range(len(self.down)):
            f_down = self.down[i](f_down)
            f_list.append(f_down)

        # Up
        f_up = f_list[-1]
        for j in range(len(self.up)):
            f_up = self.up[j](f_up, f_list[::-1][j + 1])

        # Seam head
        seam = self.head(f_up)
        seam = F.interpolate(seam, size=(or_img1.shape[2], or_img1.shape[3]))

        # Seam output
        seg1 = torch.clip(seam, 0, 1)
        m_seg1 = (mask1 - mask1 * mask2) + mask1 * mask2 * seg1
        m_seg2 = (mask2 - mask1 * mask2) + mask1 * mask2 * (1 - seg1)
        stitched_img = (or_img1 + 1.) * m_seg1 + (or_img2 + 1.) * m_seg2 - 1.
        return stitched_img, m_seg1, m_seg2


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """
        Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    or_img1 = torch.randn(1, 3, 512, 512)
    or_img2 = torch.randn(1, 3, 512, 512)
    mask1 = torch.randn(1, 3, 512, 512)
    mask2 = torch.randn(1, 3, 512, 512)
    inp = (or_img1, or_img2, mask1, mask2)
    params = {
        "in_chans_list": (8, 8, 16, 32, 64),
        "out_chans_list": (8, 16, 32, 64, 96),
        "blocks_per_stage": (2, 2, 2, 2, 2),
        "expand_ratio": (2, 2, 2, 2, 2),
        "use_attn": (False, False, False, False, True),
        "use_patchEmb": (False, True, True, True, True),
    }
    model = FastStitch(inchannels=3, inference_mode=False,
                       act_layer=nn.SiLU, **params)
    # print(model)
    # model.eval()
    # flops = FlopCountAnalysis(model, inp)
    # print(flop_count_table(flops, max_depth=5))
    res, res2, res3 = model(or_img1, or_img2, mask1, mask2)
    print("res1:", res.shape)

    # model = reparameterize_model(model)
    # flops = FlopCountAnalysis(model, inp)
    # print(flop_count_table(flops, max_depth=5))
    # res = model(inp)
    # print("res2:",res)
