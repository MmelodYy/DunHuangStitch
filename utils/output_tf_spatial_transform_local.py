# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch.nn import Upsample
import torch.nn.functional as F
import utils.tensorDLT_local as tensorDLT_local
import utils.constant as constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H

def Stitching_Domain_STN_Mesh(inputs, ortheta,  theta, size, resized_shift):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(1, n_repeats, dtype=x.dtype)
        x = torch.mm(x.reshape(-1, 1), rep)
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, channels, height, width = im.shape
        device = im.device

        x, y = x.float().to(device), y.float().to(device)
        # height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
        out_height, out_width = out_size

        # scale indices from [-1, 1] to [0, width/height]
        # effect values will exceed [-1, 1], so clamp is unnecessary or even incorrect
        # x = (x + 1.0) * width_f / 2.0
        # y = (y + 1.0) * height_f / 2.0
        # x = x / (size_tensor[0] - 1) * size_tensor[0]
        # y = y / (size_tensor[1] - 1) * size_tensor[1]

        # do sampling
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).to(device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 1).reshape(-1, channels).float()
        Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output  # .clamp(0., 1.) stupid

    # input:  batch_size*(grid_h+1)*(grid_w+1)*2
    # output: batch_size*grid_h*grid_w*9
    def get_Hs(theta, width, height):
        # print("theta:",theta.shape)
        num_batch = theta.shape[0]
        h = height / grid_h
        w = width / grid_w
        Hs = []
        for i in range(grid_h):
            for j in range(grid_w):
                hh = i * h
                ww = j * w
                ori = torch.tile(
                    torch.FloatTensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h]),
                    dims=(num_batch, 1)).to(theta.device)
                # id = i * (grid_w + 1) + grid_w
                tar = torch.cat(
                    [
                        theta[0:, i:i + 1, j:j + 1, 0:], theta[0:, i:i + 1, (j + 1):(j + 1) + 1, 0:],
                        theta[0:, (i + 1):(i + 1) + 1, j:j + 1, 0:],
                        theta[0:, (i + 1):(i + 1) + 1, (j + 1):(j + 1) + 1, 0:],
                    ], dim=1).to(theta.device)
                # print("ori:",ori.shape)
                # print("tar:", tar.shape)
                tar = tar.view(num_batch, 8)
                # tar = tf.Print(tar, [tf.slice(ori, [0, 0], [1, -1])],message="[ori--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                # tar = tf.Print(tar, [tf.slice(tar, [0, 0], [1, -1])],message="[tar--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                Hs.append(tensorDLT_local.solve_DLT(ori, tar).view(num_batch, 1, 9))
        Hs = torch.cat(Hs, dim=1).view(num_batch, grid_h, grid_w, 9)
        return Hs

    def get_Hs2(ortheta, theta):
        num_batch = theta.shape[0]
        Hs = []
        for i in range(grid_h):
            for j in range(grid_w):
                ori = torch.cat(
                    [
                        ortheta[0:, i:i + 1, j:j + 1, 0:], ortheta[0:, i:i + 1, (j + 1):(j + 1) + 1, 0:],
                        ortheta[0:, (i + 1):(i + 1) + 1, j:j + 1, 0:],
                        ortheta[0:, (i + 1):(i + 1) + 1, (j + 1):(j + 1) + 1, 0:],
                    ], dim=1).to(theta.device)
                ori = ori.view(num_batch, 8)
                tar = torch.cat(
                    [
                        theta[0:, i:i + 1, j:j + 1, 0:], theta[0:, i:i + 1, (j + 1):(j + 1) + 1, 0:],
                        theta[0:, (i + 1):(i + 1) + 1, j:j + 1, 0:],
                        theta[0:, (i + 1):(i + 1) + 1, (j + 1):(j + 1) + 1, 0:],
                    ], dim=1).to(theta.device)
                tar = tar.view(num_batch, 8)
                Hs.append(tensorDLT_local.solve_DLT(ori, tar).view(num_batch, 1, 9))
        Hs = torch.cat(Hs, dim=1).view(num_batch, grid_h, grid_w, 9)
        return Hs

    def _meshgrid(width_max, width_min, height_max, height_min, sh, eh, sw, ew):
        width, height = width_max - width_min, height_max - height_min

        hn = eh - sh + 1
        wn = ew - sw + 1

        x_t = torch.matmul(torch.ones(hn, 1), torch.transpose(torch.linspace(width_min, width_max, width.int())[sw:sw+wn].unsqueeze(1), 0, 1))
        y_t = torch.matmul(torch.linspace(height_min, height_max, height.int())[sh:sh + hn].unsqueeze(1), torch.ones(1, wn))


        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def _meshgrid2(width, height, sh, eh, sw, ew):
        hn = eh - sh + 1
        wn = ew - sw + 1

        x_t = torch.matmul(torch.ones(hn, 1),
                           torch.transpose(torch.linspace(0., float(width), width)[sw:sw+wn].unsqueeze(1), 0, 1))
        y_t = torch.matmul(torch.linspace(0., float(height), height)[sh:sh + hn].unsqueeze(1),
                           torch.ones(1, wn))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def _meshgrid3(width_max, width_min, height_max, height_min):
        width, height = int(width_max - width_min), int(height_max - height_min)
        x_t = torch.matmul(torch.ones(height, 1),
                           torch.transpose(torch.linspace(width_min, width_max, width).unsqueeze(1), 0, 1))
        y_t = torch.matmul(torch.linspace(height_min, height_max, height).unsqueeze(1),
                           torch.ones(1, width))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)

        return grid

    def _transform(image_tf, ortheta, theta , width_max, width_min, height_max, height_min):
        # print("image_tf:",image_tf.shape)
        num_batch, num_channels, height, width = image_tf.shape
        device = image_tf.device

        # grid of (x_t, y_t, 1)
        out_width = (width_max - width_min).int()
        out_height = (height_max - height_min).int()

        theta = theta.to(dtype=torch.float32)
        # Hs = get_Hs(theta, width, height)
        Hs = get_Hs2(theta,ortheta)
        # Hs = get_Hs2(ortheta,theta)
        # print("HS:",Hs.shape)
        ##########################################
        Hs = Hs.permute(0, 3, 1, 2)
        H_array = Upsample(size=(out_height, out_width), mode='nearest')(Hs)
        H_array = H_array.permute(0, 2, 3, 1)
        # print("H_array:",H_array.shape)
        H_array = H_array.view(-1, 3, 3)

        grid = _meshgrid3(width_max, width_min, height_max, height_min)
        # grid = _meshgrid3(out_width, 0, out_height, 0)
        # print("out_height:", out_height)
        # print("out_width:", out_width)
        # print("_meshgrid:", grid.shape)
        grid = grid.unsqueeze(0).to(device)
        grid = grid.view(-1)
        grid = torch.tile(grid, [num_batch])  # stack num_batch grids
        grid = grid.view(num_batch, 3, -1)
        # print("grid")
        # print(grid.shape)
        ### [bs, 3, N]

        grid = torch.transpose(grid, 1, 2).unsqueeze(3)
        # print("grid:",grid.shape)
        ### [bs, 3, N] -> [bs, N, 3] -> [bs, N, 3, 1]
        grid = grid.contiguous().view(-1, 3, 1)
        ### [bs*N, 3, 1]

        grid_row = grid.view(-1, 3)
        # print("grid_row")
        # print(grid_row.shape)
        # print(H_array[:, 0, :].shape)
        x_s = torch.sum(torch.multiply(H_array[:, 0, :], grid_row), 1)
        y_s = torch.sum(torch.multiply(H_array[:, 1, :], grid_row), 1)
        t_s = torch.sum(torch.multiply(H_array[:, 2, :], grid_row), 1)

        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        # while an affine transformation preserves it.
        t_s_flat = t_s.view(-1)
        # print("x_s:", x_s.shape)
        # print("y_s:", y_s.shape)
        # print("t_s_flat:", t_s_flat.shape)
        t_1 = torch.ones_like(t_s_flat)
        t_0 = torch.zeros_like(t_s_flat)
        sign_t = torch.where(t_s_flat >= 0, t_1, t_0) * 2 - 1
        t_s_flat = t_s_flat + sign_t * 1e-8

        x_s_flat = x_s.view(-1) / t_s_flat
        y_s_flat = y_s.view(-1) / t_s_flat


        out_size = (out_height, out_width)
        input_transformed = _interpolate(image_tf, x_s_flat, y_s_flat, out_size)

        output = input_transformed.view(num_batch, out_height, out_width, num_channels).permute(0, 3, 1, 2)

        return output



    ################################################
    ################################################
    # method 1 only can be used to global homography*
    # device = inputs.device
    # pts_1_tile = torch.tile(size, [1, 4, 1])
    # tmp = torch.FloatTensor([0., 0., 1., 0., 0., 1., 1., 1.]).unsqueeze(0).unsqueeze(2).to(device)
    # pts_1 = pts_1_tile * tmp
    # pts_2 = resized_shift + pts_1
    # pts = torch.cat((pts_1, pts_2), dim=0).reshape(8, 2)
    # pts_x, pts_y = pts.T
    # width_max, width_min, height_max, height_min = pts_x.max(), pts_x.min(), pts_y.max(), pts_y.min()
    # print("method1:",height_min, height_max, width_min, width_max)
    # method 2 mesh
    device = inputs.device
    # # print("size:",size.shape)
    img_w,img_h = size[0,0,0],size[0,1,0]
    # img_w, img_h = 512,512
    width_max = torch.max(theta[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).to(device), width_max)
    width_min = torch.min(theta[..., 0])
    width_min = torch.minimum(torch.tensor(0).to(device), width_min)
    height_max = torch.max(theta[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).to(device), height_max)
    height_min = torch.min(theta[..., 1])
    height_min = torch.minimum(torch.tensor(0).to(device), height_min)
    # print("method2:", height_min, height_max, width_min, width_max)
    ######
    # theta = torch.stack([theta[..., 0] - width_min, theta[..., 1] - height_min], 3)
    # width_max = width_max - width_min
    # height_max = height_max - height_min
    # width_min = 0
    # height_min = 0
    ######
    out_width = width_max - width_min
    out_height = height_max - height_min

    # ortheta = size
    # theta = resized_shift
    # step 1.
    # print("H_one:",H_one.shape)
    # print("theta:",theta.shape)
    img1_tf = inputs[:, 0:3, :, :]
    img1_tf = _transform(img1_tf, ortheta, ortheta, width_max, width_min, height_max, height_min)

    # step 2.
    warp_tf = inputs[:, 3:6, :, :]
    warp_tf = _transform(warp_tf, ortheta, theta, width_max, width_min, height_max, height_min)

    one = torch.ones_like(inputs[:, 0:3, :, :], dtype=torch.float32)
    mask1 = _transform(one, ortheta ,ortheta, width_max, width_min, height_max, height_min)
    mask2 = _transform(one, ortheta, theta, width_max, width_min, height_max, height_min)

    resized_height = out_height - out_height % 8
    resized_width = out_width - out_width % 8
    resized_height = int(resized_height.cpu().detach().numpy())
    resized_width = int(resized_width.cpu().detach().numpy())
    # print(img1_tf.shape)
    # print(warp_tf.shape)
    # print(mask1.shape)
    # print(mask2.shape)
    # print(resized_height,resized_width)
    img1_tf = F.interpolate(img1_tf, size=(resized_height, resized_width))
    warp_tf = F.interpolate(warp_tf, size=(resized_height, resized_width))
    mask1 = F.interpolate(mask1, size=(resized_height, resized_width))
    mask2 = F.interpolate(mask2, size=(resized_height, resized_width))

    output = torch.cat([img1_tf, warp_tf, mask1, mask2], dim=1)
    output = output.permute(0, 2, 3, 1)
    # print(output.shape)
    return output
