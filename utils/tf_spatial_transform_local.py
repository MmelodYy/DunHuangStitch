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
import numpy as np
import utils.tensorDLT_local as tensorDLT_local
import utils.constant as constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H

def transformer(U, im_one, theta,patch_size=128., name='SpatialTransformer', **kwargs):
    def _repeat(x, n_repeats):
        rep = torch.transpose(torch.ones(n_repeats).unsqueeze(1), 0, 1)
        rep = rep.to(dtype=torch.float32)
        x = x.to(dtype=torch.float32)
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch,height,width,channels = im.shape

        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        height_f = height
        width_f = width
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([], dtype=torch.int32)
        max_y = int(im.shape[1] - 1)
        max_x = int(im.shape[2] - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).to(dtype=torch.int32)
        x1 = x0 + 1
        y0 = torch.floor(y).to(dtype=torch.int32)
        y1 = y0 + 1

        x0 = torch.clip(x0, zero, max_x)
        x1 = torch.clip(x1, zero, max_x)
        y0 = torch.clip(y0, zero, max_y)
        y1 = torch.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(im.device)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.contiguous().view(-1, channels)
        im_flat = im_flat.to(dtype=torch.float32)
        # print("im_flat:",im_flat.shape)
        # print("idx_a:",idx_a.shape)
        device = im_flat.device
        # Ia = torch.gather(im_flat, 0, idx_a.unsqueeze(1).repeat(1, channels).to(device, dtype=torch.int64))
        # Ib = torch.gather(im_flat, 0, idx_b.unsqueeze(1).repeat(1, channels).to(device, dtype=torch.int64))
        # Ic = torch.gather(im_flat, 0, idx_c.unsqueeze(1).repeat(1, channels).to(device, dtype=torch.int64))
        # Id = torch.gather(im_flat, 0, idx_d.unsqueeze(1).repeat(1, channels).to(device, dtype=torch.int64))

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.to(dtype=torch.float32)
        x1_f = x1.to(dtype=torch.float32)
        y0_f = y0.to(dtype=torch.float32)
        y1_f = y1.to(dtype=torch.float32)
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1).to(device)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1).to(device)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1).to(device)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1).to(device)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output


    #input:  batch_size*(grid_h+1)*(grid_w+1)*2
    #output: batch_size*grid_h*grid_w*9
    def get_Hs(theta, patch = 128.):
        # print(theta.device)
        num_batch = theta.shape[0]
        h = patch / grid_h
        w = patch / grid_w
        Hs = []
        for i in range(grid_h):
            for j in range(grid_w):
                hh = i * h
                ww = j * w
                ori = torch.tile(
                    torch.FloatTensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h]),
                    dims=(num_batch, 1)).to(theta.device)
                #id = i * (grid_w + 1) + grid_w
                tar = torch.cat(
                    [
                        theta[0:, i:i + 1, j:j + 1, 0:],
                        theta[0:, i:i + 1, (j + 1):(j + 1) + 1, 0:],
                        theta[0:, (i + 1):(i + 1) + 1, j:j + 1, 0:],
                        theta[0:, (i + 1):(i + 1) + 1, (j + 1):(j + 1) + 1, 0:],
                    ], dim=1).to(theta.device)
                tar = tar.view(num_batch, 8)
                Hs.append(tensorDLT_local.solve_DLT(ori, tar).view(num_batch, 1, 9))
        Hs = torch.cat(Hs, dim=1).view(num_batch, grid_h, grid_w, 9)
        return Hs

    def _meshgrid2(height, width):
        x_t = torch.matmul(torch.ones(height, 1),
                        torch.transpose(torch.linspace(-1.0, 1.0, width).unsqueeze(1), 0, 1))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones(1, width))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid


    def _transform3(theta, input_dim, im_one, patch_size = 128.):
        # print("theta:", theta.shape)
        # print("input_dim:", input_dim.shape)
        # print("im_one:", im_one.shape)
        # print("depth:", depth.shape)
        num_batch, height, width, num_channels  = input_dim.shape
        device = theta.device
        # patch_size = 128.
        M = torch.FloatTensor([[patch_size / 2.0, 0., patch_size / 2.0],
              [0., patch_size / 2.0, patch_size / 2.0],
              [0., 0., 1.]]).to(device)
        M_tensor = M.to(dtype= torch.float32)
        M_tile = torch.tile(M_tensor.unsqueeze(0), [num_batch*grid_h*grid_w, 1, 1])
        M_inv = torch.linalg.inv(M)
        M_tensor_inv = M_inv.to(dtype= torch.float32)
        M_tile_inv = torch.tile(M_tensor_inv.unsqueeze(0), [num_batch*grid_h*grid_w, 1, 1])
        # print("M_tile_inv:",M_tile_inv.shape)
        # print("M_tile:", M_tile.shape)
        theta = theta.to(dtype= torch.float32)
        Hs = get_Hs(theta, patch_size)
        Hs = Hs.view(-1, 3, 3)
        Hs = torch.matmul(torch.matmul(M_tile_inv, Hs), M_tile)
        Hs = Hs.view(num_batch, grid_h, grid_w, 9)

        Hs = Hs.permute(0, 3, 1, 2)
        H_array = Upsample(size=(height, width), mode='nearest')(Hs)
        H_array = H_array.permute(0, 2, 3, 1)
        # print("H_array:",H_array.shape)
        H_array = H_array.view(-1, 3, 3)
        ##########################################

        out_height = height
        out_width = width
        grid = _meshgrid2(out_height, out_width)
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


        out_size = (height, width)
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        mask_transformed = _interpolate(im_one, x_s_flat, y_s_flat, out_size)

        output = input_transformed.view(num_batch, height, width, num_channels)
        mask_output = mask_transformed.view(num_batch, height, width, num_channels)

        # print("!@#$%^===output/black_pix=======================")
        # print(output)
        # print("!@#$%^==========================")
        return output, mask_output

    U = U.permute(0, 2, 3, 1)
    im_one = im_one.permute(0, 2, 3, 1)
    output, mask_output = _transform3(theta, U, im_one, patch_size)
    output = output.permute(0, 3, 1, 2)
    mask_output = mask_output.permute(0, 3, 1, 2)
    return output, mask_output



