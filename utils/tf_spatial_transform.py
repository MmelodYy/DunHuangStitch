import torch


def STN(image2_tensor, H_tf):
    """Spatial Transformer Layer"""
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
        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0

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

    def _meshgrid(height, width):
        x_t = torch.mm(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).unsqueeze(0))
        y_t = torch.mm(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones(1, width))

        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid

    bs, nc, height, width = image2_tensor.shape
    device = image2_tensor.device

    H_tf = H_tf.reshape(-1, 3, 3).float()
    # grid of (x_t, y_t, 1)
    grid = _meshgrid(height, width).unsqueeze(0).expand(bs, -1, -1).to(device)

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

    input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height, width))
    output = input_transformed.reshape(bs, height, width, nc).permute(0, 3, 1, 2)
    return output



