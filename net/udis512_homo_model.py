import torch
import numpy as np
from torch import nn
import torch.nn.functional
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from utils.tensorDLT import solve_DLT
from utils.tf_spatial_transform import STN
from utils.output_tensorDLT import solve_Size_DLT
from utils.output_tf_spatial_transform import Stitching_Domain_STN

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.GELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad2(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.norm1 = nn.LayerNorm(c2)
        self.out_dim = c2

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def extract_patches(x, kernel=3, stride=1):
    if kernel != 1:
        x = nn.ZeroPad2d(1)(x)
    # print("x:",x.shape)
    x = x.permute(0, 2, 3, 1)
    # print("x:", x.shape)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    # print("all_patches:", all_patches.shape)
    return all_patches


def CCL(feature_1, feature_2,search_range=16., normBoth=False):
    device = feature_1.device
    # print("feature_1:",feature_1.shape)
    # print("feature_2:", feature_2.shape)
    bs, c, h, w = feature_1.size()
    if normBoth:
        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
    else:
        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = feature_2

    # print(norm_feature_2.size())
    # print("norm_feature_2:", norm_feature_2.shape)
    patches = extract_patches(norm_feature_2)
    # print(patches.shape)
    patches = patches.to(device)
    matching_filters = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))
    # print(matching_filters.shape)
    # if normBoth:
    #     norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
    #     norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
    # else:
    #     norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
    # bs, c, h, w = feature_1.shape
    # padded_x2 = F.pad(norm_feature_2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
    # max_offset = search_range * 2 + 1
    #
    # # faster(*2) but cost higher(*n) GPU memory
    # matching_filters = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)

    match_vol = []
    for i in range(bs):
        single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
        match_vol.append(single_match)

    match_vol = torch.cat(match_vol, 0)
    # print(match_vol .size())

    # scale softmax
    softmax_scale = 10
    match_vol = F.softmax(match_vol * softmax_scale, 1)

    channel = match_vol.size()[1]

    h_one = torch.linspace(0, h - 1, h).to(device)
    one1w = torch.ones(1, w).to(device)
    h_one = torch.matmul(h_one.unsqueeze(1), one1w)
    h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)
    w_one = torch.linspace(0, w - 1, w).to(device)
    oneh1 = torch.ones(h, 1).to(device)
    w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
    w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

    c_one = torch.linspace(0, channel - 1, channel).to(device)
    c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

    flow_h = match_vol * (c_one // w - h_one)
    flow_h = torch.sum(flow_h, dim=1, keepdim=True)
    flow_w = match_vol * (c_one % w - w_one)
    flow_w = torch.sum(flow_w, dim=1, keepdim=True)

    feature_flow = torch.cat([flow_w, flow_h], 1)
    # print(feature_flow.size())

    return feature_flow



def cost_volume(x1, x2, search_range, normBoth=False, fast=True):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
      Args:
          c1: Level of the feature pyramid of Image1
          warp: Warped level of the feature pyramid of image22
          search_range: Search range (maximum displacement)
      """
    if normBoth:
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
    else:
        x1 = F.normalize(x1, p=2, dim=1)
    bs, c, h, w = x1.shape
    padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
    max_offset = search_range * 2 + 1

    if fast:
        # faster(*2) but cost higher(*n) GPU memory
        patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
        # print(patches.shape)
        cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
    else:
        # slower but save memory
        cost_vol = []
        for j in range(0, max_offset):
            for i in range(0, max_offset):
                x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                cost_vol.append(cost)
        cost_vol = torch.cat(cost_vol, dim=1)

    cost_vol = F.leaky_relu(cost_vol, 0.1)
    return cost_vol


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        ##########################
        mlp_ratio = 2.
        drop = 0
        drop_path = 0.2
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=nn.GELU,
                       drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None, dw=None):
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if dw is not None:
            x = x + dw
        # x = self.proj(x)
        # x = self.proj_drop(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MSABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)


    def forward(self, x, mask_matrix):

        B, H, W, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert H * W == self.input_resolution[0] * self.input_resolution[1], "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        dw = shifted_x.permute(0, 3, 1, 2).contiguous()
        dw = self.dwconv(dw)
        dw = dw.permute(0, 2, 3, 1).contiguous()
        dw = window_partition(dw, self.window_size)  # nW*B, window_size, window_size, C
        dw = dw.view(-1, self.window_size * self.window_size,
                     C)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, dw=dw)  # nW*B, window_size*window_size, C
        # attn_windows = self.attn(x_windows, mask=attn_mask, dw=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # x = x.view(B,  H * W, C)
        x = shortcut + self.drop_path(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            # print(x.shape)
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(self.contract(x))

    @staticmethod
    def contract(x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block = int(np.log2(patch_size[0]))
        self.project_block = []
        self.dim = [int(embed_dim) // (2 ** i) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim = self.dim[::-1]  # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim

        self.focus = Focus(self.dim[0], self.dim[-1], 3, 1)


    def forward(self, x):
        x = self.focus(x)
        return x


class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=[224, 224],
                 patch_size=[4, 4],
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 3, 3, 3],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # print("xxxxx：",pretrain_img_size[0] // patch_size[0] // 2 ** i_layer)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (i_layer-1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (i_layer-1)),
                    # pretrain_img_size[0] // patch_size[0] // 2 ** (i_layer),
                    # pretrain_img_size[1] // patch_size[1] // 2 ** (i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""
        down = []
        x = self.patch_embed(x)
        # print(x.shape)
        Wh, Ww = x.size(2), x.size(3)

        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = x_out.permute(0, 2, 3, 1)
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                down.append(out)
        return down


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, input_resolution=None, num_heads=None,
                 window_size=None, i_block=None, qkv_bias=None, qk_scale=None):
        super().__init__()

        self.blocks_tr = MSABlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (i_block % 2 == 0) else window_size // 2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=0,
            attn_drop=0,
            drop_path=drop_path)

    def forward(self, x, mask):
        x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape,mask.shape)
        x = self.blocks_tr(x, mask)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class FeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_channels,
                 embedding_dim,
                 depths,
                 num_heads,
                 num_classes,
                 crop_size,
                 patch_size,
                 window_size,
                 out_indices):
        super(FeatureExtractor, self).__init__()

        # Don't uncomment conv_op
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.embed_dim = embedding_dim
        self.depths = depths
        self.num_heads = num_heads
        self.crop_size = [crop_size, crop_size]
        self.patch_size = [patch_size, patch_size]
        self.window_size = window_size
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1
        # my love shq and xzh and dy and zty
        self.model_down = encoder(
            pretrain_img_size=self.crop_size,
            window_size=self.window_size,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            depths=self.depths,
            num_heads=self.num_heads,
            in_chans=self.num_input_channels,
            out_indices=out_indices
        )

    def forward(self, x):
        skips = self.model_down(x)
        return skips


class RegressionNet(nn.Module):
    def __init__(self,fIndim=256,
                 pretrain_img_size=[64, 64],
                 embed_dim=16,
                 depths=[3, 3, 3, 3],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super(RegressionNet, self).__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # print("i_layer：", i_layer, "dim:", embed_dim * 2 ** i_layer)
            # print("i_layer：",i_layer,"input_resolution:",pretrain_img_size[0] // 2 ** i_layer)
            # print("i_layer：", i_layer, "input_resolution:", pretrain_img_size[1] // patch_size[1] // 2 ** i_layer)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // 2 ** (i_layer),
                    pretrain_img_size[1]  // 2 ** (i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i)*2 for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # print(num_features[-1])
        self.fc = nn.Sequential(
            Conv(num_features[-1], 64, 3, 1),
            Conv(64, 64, 3, 2),
            nn.Flatten(),
            nn.Linear(fIndim,1024),
            # nn.SiLU(inplace=True),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(1024,8)
        )

    def forward(self,x):
        Wh, Ww = x.size(2), x.size(3)
        regdown = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            # print("xx:",x.shape)
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # print("x:", x.shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = x.permute(0, 2, 3, 1)
                x_out = norm_layer(x_out)

                out = x_out.view(-1, Wh, Ww, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                regdown.append(out)
        # print("regdown:",regdown[-1].shape)
        x = self.fc(regdown[-1])
        return x


class AttentionCostVolume(nn.Module):
    def __init__(self,search_range=8):
        super(AttentionCostVolume, self).__init__()
        max_offset = search_range * 2 + 1
        in_planes = max_offset*max_offset
        out_planes = 49
        self.att = nn.Conv2d(in_planes, in_planes, kernel_size=7, padding=3, groups=in_planes)
        # self.att = EMA2(in_planes)
        self.agg = nn.Sequential(
            Conv(in_planes,in_planes//2,3,1),
            Conv(in_planes//2,out_planes,3,1)
            # Conv(in_planes,out_planes,3,1)
        )
        self.sr = search_range

    def forward(self,f1,f2,normBoth=False):
        match_vol = cost_volume(f1,f2, self.sr,normBoth=normBoth)
        # print("match_vol:",match_vol.shape)
        att_vol = match_vol * self.att(match_vol)
        # print("att_vol:",att_vol.shape)
        agg_vol = self.agg(att_vol)
        return agg_vol


class HModel(nn.Module):
    def __init__(self,inplanes):
        super(HModel, self).__init__()
        self.feature_extractor =  FeatureExtractor(num_input_channels=inplanes,
                 embedding_dim=16,
                 depths=[2, 2, 2],
                 num_heads=[8, 8, 8],
                 num_classes=2,
                 crop_size=128,
                 patch_size=4,
                 window_size=[4, 4, 4],
                 out_indices=(0, 1, 2))
        print("---------------reg1----------------------")
        self.corr1 = AttentionCostVolume(16)
        self.reg1 = RegressionNet(pretrain_img_size=[16, 16],
                 embed_dim=49,
                 depths=[2],
                 num_heads=[7],
                 window_size=[4],
                 out_indices=(0,),
                 fIndim=1024)
        print("---------------reg2----------------------")
        self.corr2 = AttentionCostVolume(8)
        self.reg2 = RegressionNet(pretrain_img_size=[32, 32],
                  embed_dim=49,
                  depths=[2, 2],
                  num_heads=[7, 7],
                  window_size=[4, 4],
                  out_indices=(0, 1),
                  fIndim=1024)
        print("---------------reg3----------------------")
        self.corr3 = AttentionCostVolume(4)
        self.reg3 = RegressionNet(pretrain_img_size=[64, 64],
                 embed_dim=49,
                 depths=[2, 2, 2],
                 num_heads=[7, 7, 7],
                 window_size=[4, 4, 4],
                 out_indices=(0, 1, 2),
                 fIndim=1024)

        self.DLT_solver = solve_DLT()
        self.DLT_Size_solver = solve_Size_DLT()

    def DLT_transform(self,img, net1_f, patch_size=32., s=4.,isNorm=True):
        bs = net1_f.shape[0]
        _, _, h, w = img.shape
        device = net1_f.device
        H1_inv = self.DLT_solver.solve(net1_f / s, patch_size)
        M = torch.FloatTensor([[patch_size / 2.0, 0., patch_size / 2.0],
                               [0., patch_size / 2.0, patch_size / 2.0],
                               [0., 0., 1.]]).to(device)
        M_inv = torch.linalg.inv(M)
        H1_mat = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H1_inv), M.expand(bs, -1, -1))

        if isNorm:
            feature2_warp = STN(F.normalize(img, p=2, dim=1), H1_mat)
        else:
            feature2_warp = STN(img, H1_mat)
        return feature2_warp,H1_mat

    def forward_once(self, x1, x2):
        b, c, _, _ = x1.shape
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        # print("feature1:", f1[-1].shape)
        # print("feature2:", f1[-2].shape)
        # print("feature3:", f1[-3].shape)
        ############# regressionNet 1 #############
        search_range = 3
        patch_size = 32.
        stride = 4.
        # global_correlation = cost_volume(f1[-1],f2[-1], search_range,normBoth=True)
        # global_correlation = CCL(f1[-1], f2[-1], normBoth=True)
        global_correlation = self.corr1(f1[-1], f2[-1], normBoth=True)
        # print("global_correlation:",global_correlation.shape)
        net1_fc2 = self.reg1(global_correlation)
        net1_f = net1_fc2.unsqueeze(2)
        feature2_warp,_ = self.DLT_transform(f2[-2], net1_f, patch_size, stride,isNorm=True)
        # print("feature2_warp:",feature2_warp.shape)
        ############# regressionNet 2 #############
        search_range = 3
        patch_size = 64.
        stride = 2.
        # local_correlation_2 = cost_volume(f1[-2], feature2_warp, search_range,normBoth=False)
        # local_correlation_2 = CCL(f1[-2], feature2_warp, normBoth=False)
        local_correlation_2 = self.corr2(f1[-2], feature2_warp, normBoth=False)
        # print("local_correlation_2:",local_correlation_2.shape)
        net2_fc2 = self.reg2(local_correlation_2)
        net2_f = net2_fc2.unsqueeze(2)
        feature3_warp,_ = self.DLT_transform(f2[-3], net1_f + net2_f, patch_size, stride,isNorm=True)
        # print("feature3_warp:",feature3_warp.shape)
        ############# regressionNet 3 #############
        search_range = 3
        # local_correlation_3 = cost_volume(f1[-3], feature3_warp, search_range,normBoth=False)
        # local_correlation_3 = CCL(f1[-3], feature3_warp,  normBoth=False)
        local_correlation_3 = self.corr3(f1[-3], feature3_warp, normBoth=False)
        # print("local_correlation_3:",local_correlation_3.shape)
        net3_fc2 = self.reg3(local_correlation_3)
        net3_f = net3_fc2.unsqueeze(2)
        return net1_f, net2_f, net3_f

    def output_H_estimator(self,or_input1, or_input2,input1, input2,size,imgsz=128.):
        net1_f, net2_f, net3_f = self.forward_once(input1, input2)
        shift = net1_f + net2_f + net3_f
        # print(shift)
        # print(size.shape)
        # print(shift.shape)
        size = size.unsqueeze(2)
        size_tmp = torch.cat([size, size, size, size], dim=1) / imgsz
        resized_shift = torch.mul(shift, size_tmp)
        # print(size)
        # print("size:", size.shape)
        # print("shift:", shift.shape)
        # print("size_tmp:",size_tmp.shape)
        # print("resized_shift:", resized_shift.shape)
        H = self.DLT_Size_solver.solve(resized_shift, size)
        # print("H:",H)
        # H = ''
        # size = size.squeeze(0).squeeze(1)
        # resized_shift = shift * size.repeat(4).reshape(1, 8, 1) / imgsz
        coarsealignment = Stitching_Domain_STN(torch.cat([or_input1, or_input2],dim=1), H, size, resized_shift)
        return coarsealignment

    def forward(self,inputs_aug1,inputs_aug2, input1,inputs2, patch_size=128.):
        # inputs_aug1 = self.ShareFeature(inputs_aug1)
        # inputs_aug2 = self.ShareFeature(inputs_aug2)
        net1_f, net2_f, net3_f = self.forward_once(inputs_aug1, inputs_aug2)

        stride = 1.
        warp2_H1, H1_mat = self.DLT_transform(inputs2, net1_f, patch_size, stride, isNorm=False)
        warp2_H2, H2_mat = self.DLT_transform(inputs2, net1_f + net2_f, patch_size, stride, isNorm=False)
        warp2_H3, H3_mat = self.DLT_transform(inputs2, net1_f + net2_f + net3_f, patch_size, stride, isNorm=False)

        one = torch.ones_like(inputs2, dtype=torch.float32)
        one_warp_H1 = STN(one, H1_mat)
        one_warp_H2 = STN(one, H2_mat)
        one_warp_H3 = STN(one, H3_mat)

        # H1_inv = self.DLT_solver.solve(net1_f + net2_f + net3_f, patch_size)
        # print("H1_inv:",H1_inv)
        # print("net1_f + net2_f + net3_f:",net1_f + net2_f + net3_f)
        return net1_f, net2_f, net3_f, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3

