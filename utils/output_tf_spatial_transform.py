import cv2
import torch
import torch.nn.functional as F

def Stitching_Domain_STN(inputs, H_tf, size, resized_shift):
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
        height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
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
        # Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch,channels)
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

    def _meshgrid(width_max, width_min, height_max, height_min):
        width, height = width_max - width_min, height_max - height_min
        x_t = torch.mm(torch.ones(height.int(), 1), torch.linspace(width_min, width_max, width.int()).unsqueeze(0))
        y_t = torch.mm(torch.linspace(height_min, height_max, height.int()).unsqueeze(1), torch.ones(1, width.int()))

        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid

    def _transform(image_tf, H_tf, width_max, width_min, height_max, height_min):
        bs, nc, height, width = image_tf.shape
        device = image_tf.device

        H_tf = H_tf.reshape(-1, 3, 3).float()
        # grid of (x_t, y_t, 1)
        out_width = (width_max - width_min).int()
        out_height = (height_max - height_min).int()
        grid = _meshgrid(width_max, width_min, height_max, height_min).unsqueeze(0).expand(bs, -1, -1).to(device)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = torch.bmm(H_tf, grid)  # [bs,3,3] x [bs,3,w*h] -> [bs,3,w*h]
        x_s, y_s, t_s = torch.chunk(T_g, 3, dim=1)
        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = t_s.reshape(-1)
        eps, maximal = 1e-2, 10.
        t_s_flat[t_s_flat.abs() < eps] = eps
        # 1.25000 / 1.38283e-05 = inf   in float16 (6.55e4)

        #  batchsize * width * height
        x_s_flat = x_s.reshape(-1) / t_s_flat
        y_s_flat = y_s.reshape(-1) / t_s_flat

        input_transformed = _interpolate(image_tf, x_s_flat, y_s_flat, (out_height, out_width))

        output = input_transformed.reshape(bs, out_height, out_width, nc).permute(0, 3, 1, 2)

        return output

    ################################################
    ################################################
    ################################################
    device = inputs.device
    pts_1_tile = torch.tile(size, [1, 4, 1])
    tmp = torch.FloatTensor([0., 0., 1., 0., 0., 1., 1., 1.]).unsqueeze(0).unsqueeze(2).to(device)
    pts_1 = pts_1_tile * tmp
    # print(pts_1)
    # print(resized_shift)
    pts_2 = resized_shift + pts_1


    # # print("pts_1:",pts_1)
    # # print("pts_2:",pts_2)
    # width_list1 = [pts_1[:,i,:].unsqueeze(1) for i in range(0, 8, 2)]
    # width_list2 = [pts_2[:,i,:].unsqueeze(1) for i in range(0, 8, 2)]
    # # print(width_list1[0].shape)
    # height_list1 = [pts_1[:,i,:].unsqueeze(1) for i in range(1, 8, 2)]
    # height_list2 = [pts_2[:,i,:].unsqueeze(1) for i in range(1, 8, 2)]
    # width_list = width_list1 + width_list2
    # height_list = height_list1 + height_list2
    # # print(width_list)
    # width_list_tf = torch.cat(width_list, dim=1)
    # height_list_tf = torch.cat(height_list, dim=1)
    # width_max = width_list_tf.max()
    # width_min = width_list_tf.min()
    # height_max = height_list_tf.max()
    # height_min = height_list_tf.min()
    # # print("height_min")
    # # print(height_min.shape)
    # out_width = width_max - width_min
    # out_height = height_max - height_min

    pts = torch.cat((pts_1, pts_2), dim=0).reshape(8, 2)
    pts_x, pts_y = pts.T
    width_max, width_min, height_max, height_min = pts_x.max(), pts_x.min(), pts_y.max(), pts_y.min()
    out_width = width_max - width_min
    out_height = height_max - height_min


    # print(width_max, width_min, height_max, height_min)
    batch_size = inputs.shape[0]
    # step 1.
    H_one = torch.eye(3)
    H_one = torch.tile(H_one.unsqueeze(0), [batch_size, 1, 1]).to(device)
    img1_tf = inputs[:,0:3,:,:]
    img1_tf = _transform(img1_tf, H_one, width_max, width_min, height_max, height_min)

    # step 2.
    warp_tf = inputs[:,3:6,:,:]
    warp_tf = _transform(warp_tf, H_tf, width_max, width_min, height_max, height_min)

    one = torch.ones_like(inputs[:,0:3,:,:], dtype=torch.float32)
    mask1 = _transform(one, H_one, width_max, width_min, height_max, height_min)
    mask2 = _transform(one, H_tf, width_max, width_min, height_max, height_min)

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






