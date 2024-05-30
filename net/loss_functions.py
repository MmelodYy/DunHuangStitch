import numpy as np
# L1 weight
import torch.nn as nn
import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import utils.constant as constant
device  =constant.GPU
grid_h = constant.GRID_H
grid_w = constant.GRID_W
img_H = constant.Img_H
img_W = constant.Img_W

min_w = (img_H / grid_w) / 8
min_h = (img_W / grid_h) / 8

# intra-grid constraint
def intra_grid_loss(pts):
    batch_size = pts.shape[0]

    delta_x = pts[:, :, 0:grid_w, 0] - pts[:, :, 1:grid_w + 1, 0]
    delta_y = pts[:, 0:grid_h, :, 1] - pts[:, 1:grid_h + 1, :, 1]

    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + min_h)

    loss = torch.mean(loss_x) + torch.mean(loss_y)
    return loss


# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:, :, 0:grid_w, :] - train_mesh[:, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / \
            (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 0:grid_w - 1, :], 3))
             * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    # print("cos_w.shape")
    # print(cos_w.shape)
    delta_w_angle = 1 - cos_w

    h_edges = train_mesh[:, 0:grid_h, :, :] - train_mesh[:, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / \
            (torch.sqrt(torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 0:grid_h - 1, :, :], 3))
             * torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    delta_h_angle = 1 - cos_h

    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    return loss

class Homo_Mesh_Total_Loss(nn.Module):
    def __init__(self,l_num):
        super().__init__()
        self.loss = nn.L1Loss()#
        if l_num == 2:
            self.loss = nn.MSELoss()

    def forward(self, mesh, img1,train_warp2_H1, train_warp2_H2, train_warp2_H3, \
            train_one_warp_H1, train_one_warp_H2, train_one_warp_H3):
        eps = 1e-12
        loss1 = self.loss(train_warp2_H1, img1*train_one_warp_H1)
        loss2 = self.loss(train_warp2_H2, img1*train_one_warp_H2)
        loss3 = self.loss(train_warp2_H3, img1*train_one_warp_H3)
        mesh_loss = 16.0*inter_grid_loss(mesh) + 16.0*intra_grid_loss(mesh)
        loss_lp = 1.0*loss1 + 4.0*loss2 + 16.0*loss3
        total_loss = loss_lp +  mesh_loss
        return total_loss, loss_lp, mesh_loss


class Homo_Total_Loss(nn.Module):
    def __init__(self,l_num):
        super().__init__()
        self.loss = nn.L1Loss()#
        if l_num == 2:
            self.loss = nn.MSELoss()

    def forward(self,img1,train_warp2_H1, train_warp2_H2, train_warp2_H3, \
            train_one_warp_H1, train_one_warp_H2, train_one_warp_H3):
        eps = 1e-12
        loss1 = self.loss(train_warp2_H1, img1*train_one_warp_H1)
        loss2 = self.loss(train_warp2_H2, img1*train_one_warp_H2)
        loss3 = self.loss(train_warp2_H3, img1*train_one_warp_H3)
        loss_lp = 1.0*loss1 + 4.0*loss2 + 16.0*loss3
        return loss_lp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class PatchFilter(nn.Module):
    def __init__(self,M=9):
        super().__init__()
        self.device = device
        # self.kernel = torch.FloatTensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).to(self.device)
        self.kernel = torch.ones((3,1,M,M)).to(self.device)

    def forward(self,img):
        b2,c2,h2,w2 = img.shape
        # print(img.shape)
        patch =  F.conv2d(img, self.kernel,padding=4,groups=c2)
        return patch


def edge_extraction(gen_frames):
    b, c, h, w = gen_frames.shape
    pos = torch.FloatTensor(np.identity((c))).to(device)
    # print(pos.shape)
    neg = -1 * pos
    filter_x = torch.stack([neg,pos]).unsqueeze(1) # [-1, 1]
    filter_y = torch.stack([pos.unsqueeze(1), neg.unsqueeze(1)])  # [[1],[-1]]
    # print(filter_x.shape)
    gen_dx = torch.abs(F.conv2d(gen_frames, filter_x,padding=1,groups=c))
    gen_dy = torch.abs(F.conv2d(gen_frames, filter_y, padding=1,groups=c))
    # print(gen_dy.shape)+1
    # edge = gen_dx ** 1 + gen_dy ** 1
    edge = torch.abs(gen_dx[:,0,:,:] - gen_dx[:,1,:,:]) + torch.abs(gen_dy[:,0,:,:] - gen_dy[:,1,:,:])
    edge_clip = torch.clip_(edge, 0, 1)
    print(edge_clip.shape)
    # condense into one tensor and avg
    return edge_clip


def seammask_extraction(mask):
    b, c, h, w = mask.shape
    seam_mask = edge_extraction(torch.mean(mask, dim=1).unsqueeze(1))
    filters = torch.FloatTensor([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]).to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    test_conv1 =F.conv2d(seam_mask,filters,padding=1,groups=c)
    test_conv1 = torch.clip_(test_conv1, 0, 1)
    test_conv2 =F.conv2d(test_conv1,filters,padding=1,groups=c)
    test_conv2 = torch.clip_(test_conv2, 0, 1)
    test_conv3 =F.conv2d(test_conv2,filters,padding=1,groups=c)
    test_conv3 = torch.clip_(test_conv3, 0, 1)
    # condense into one tensor and avg
    return test_conv3


class SeamMaskExtractor(object):
    def __init__(self, device):
        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=np.float32)
        sobel_y = np.array([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], dtype=np.float32)
        ones = np.ones((3, 3), dtype=np.float32)
        kernels = []
        for kernel in [sobel_x, sobel_y, ones]:
            kernel = np.reshape(kernel, (1, 1, 3, 3))
            kernels.append(torch.from_numpy(kernel).to(device=device))
        self.edge_kernel_x, self.edge_kernel_y, self.seam_kernel = kernels

    @torch.no_grad()
    def __call__(self, mask,isEdge=False):
        # shape(b,1,h,w)
        # print("mask:",mask.shape)
        assert isinstance(mask, torch.Tensor) and len(mask.shape) == 4 and mask.size(1) == 1
        if self.edge_kernel_x.dtype != mask.dtype:
            self.edge_kernel_x = self.edge_kernel_x.type_as(mask)
            self.edge_kernel_y = self.edge_kernel_y.type_as(mask)
            self.seam_kernel = self.seam_kernel.type_as(mask)

        mask_dx = F.conv2d(mask, self.edge_kernel_x, bias=None, stride=1, padding=1).abs()
        mask_dy = F.conv2d(mask, self.edge_kernel_y, bias=None, stride=1, padding=1).abs()
        if isEdge:
            return mask_dx+mask_dy
        else:
            edge = (mask_dx + mask_dy).clamp_(0, 1)
            for _ in range(3):  # dilate
                edge = F.conv2d(edge, self.seam_kernel, bias=None, stride=1, padding=1).clamp_(0, 1)

            return edge

class GradfExtractor(object):
    def __init__(self, device):
        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=np.float32)
        sobel_y = np.array([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], dtype=np.float32)
        kernels = []
        for kernel in [sobel_x, sobel_y]:
            kernel = np.reshape(kernel, (1, 1, 3, 3))
            kernels.append(torch.from_numpy(kernel).to(device))
        self.edge_kernel_x, self.edge_kernel_y = kernels

    @torch.no_grad()
    def __call__(self, mask):
        assert isinstance(mask, torch.Tensor) and len(mask.shape) == 4 and mask.size(1) == 1
        if self.edge_kernel_x.dtype != mask.dtype:
            self.edge_kernel_x = self.edge_kernel_x.type_as(mask)
            self.edge_kernel_y = self.edge_kernel_y.type_as(mask)

        mask_dx = F.conv2d(mask, self.edge_kernel_x, bias=None, stride=1, padding=1).abs()
        mask_dy = F.conv2d(mask, self.edge_kernel_y, bias=None, stride=1, padding=1).abs()
        return mask_dx+mask_dy

def boundary_extraction(mask, in_channel = 1, out_channel = 1):

    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    #define kernel
    kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel,in_channel,3,3)
    if torch.cuda.is_available():
        kernel = kernel.to(device)
        ones = ones.to(device)
        zeros = zeros.to(device)
    weight = nn.Parameter(data=kernel, requires_grad=False)

    #dilation
    x = F.conv2d(1-mask,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)

    # return x*mask
    return x

def intensity_loss(gen_frames, gt_frames, l_num=1):
    return torch.abs((gen_frames - gt_frames) ** l_num)

# class FuseLoss(nn.Module):
#     def __init__(self):
#         super(FuseLoss, self).__init__()
#         self.ssim_loss = SSIM()
#         self.l1_loss = nn.L1Loss()
#         self.seam_extractor = SeamMaskExtractor(device)

#     def __call__(self, image1, image2, mask1,mask2, stitched_img, seg1, seg2):
#         a = 10000
#         b = 1000
#         eps = 0.01
#         mask1, mask2 = (mask1 > eps).int().type_as(image1), (mask2 > eps).int().type_as(image1)  # binarize
#         m1 = mask1 * self.seam_extractor(mask2[:,0,:,:].float().unsqueeze(1))
#         m2 = mask2 * self.seam_extractor(mask1[:,0,:,:].float().unsqueeze(1))

#         l_boundary1 = intensity_loss(stitched_img*m1 , image1*m1, l_num=1)
#         l_boundary2 = intensity_loss(stitched_img*m2 , image2*m2, l_num=1)
#         l_boundary = torch.mean(l_boundary1) + torch.mean(l_boundary2)

#         Id = intensity_loss(image1 , image2, l_num=2)
#         ld1 =  intensity_loss(seg1[:,:,0:-1,:] , seg1[:,:,1:,:] ,l_num=1) * (Id[:,:,0:-1,:] + Id[:,:,1:,:])
#         ld2 = intensity_loss(seg1[:,:,:,0:-1] , seg1[:,:,:,1:] ,l_num=1) * (Id[:,:,:,0:-1] + Id[:,:,:,1:])
#         l_d = torch.mean(ld1) + torch.mean(ld2)

#         Is1 = intensity_loss(seg1[:,:,0:-1,:] , seg1[:,:,1:,:] ,l_num=1) * intensity_loss(stitched_img[:,:,0:-1,:], stitched_img[:,:,1:,:], l_num=1)
#         Is2 = intensity_loss(seg1[:,:,:,0:-1] , seg1[:,:,:,1:] ,l_num=1) * intensity_loss(stitched_img[:,:,:,0:-1], stitched_img[:,:,:,1:], l_num=1)
#         l_s = torch.mean(Is1) + torch.mean(Is2)

#         l_total = a*l_boundary + b*(l_d + l_s)
#         return l_total,a*l_boundary,b*(l_d + l_s)

class FuseLoss(nn.Module):
    def __init__(self):
        super(FuseLoss, self).__init__()
        self.ssim_loss = SSIM()
        self.l1_loss = nn.L1Loss()
        self.seam_extractor = SeamMaskExtractor(device)
        # self.grad_extractor = GradfExtractor(device)

    def __call__(self, image1, image2, mask1,mask2, stitched_img, seg1, seg2):
        a = 10000
        b = 1000
        eps = 0.01
        mask1, mask2 = (mask1 > eps).int().type_as(image1), (mask2 > eps).int().type_as(image1)  # binarize
        m1 = mask1 * self.seam_extractor(mask2[:,0,:,:].float().unsqueeze(1))
        m2 = mask2 * self.seam_extractor(mask1[:,0,:,:].float().unsqueeze(1))

        l_boundary1 = intensity_loss(stitched_img*m1 , image1*m1, l_num=1)
        l_boundary2 = intensity_loss(stitched_img*m2 , image2*m2, l_num=1)
        l_boundary = torch.mean(l_boundary1) + torch.mean(l_boundary2)

        Id = intensity_loss(image1 , image2, l_num=2)
        ld1 =  intensity_loss(seg1[:,:,0:-1,:] , seg1[:,:,1:,:] ,l_num=1) * (Id[:,:,0:-1,:] + Id[:,:,1:,:])
        ld2 = intensity_loss(seg1[:,:,:,0:-1] , seg1[:,:,:,1:] ,l_num=1) * (Id[:,:,:,0:-1] + Id[:,:,:,1:])
        l_d = torch.mean(ld1) + torch.mean(ld2)

        # print("l_d:",l_d.max())

        Is1 = intensity_loss(seg1[:,:,0:-1,:] , seg1[:,:,1:,:] ,l_num=1) * intensity_loss(stitched_img[:,:,0:-1,:], stitched_img[:,:,1:,:], l_num=1)
        Is2 = intensity_loss(seg1[:,:,:,0:-1] , seg1[:,:,:,1:] ,l_num=1) * intensity_loss(stitched_img[:,:,:,0:-1], stitched_img[:,:,:,1:], l_num=1)
        l_s = torch.mean(Is1) + torch.mean(Is2)

        edge1 = self.seam_extractor(image1[:,0,:,:].float().unsqueeze(1),isEdge=True)/255. #.squeeze(1).permute(1,2,0).cpu().detach().numpy()
        edge2 = self.seam_extractor(image2[:,0,:,:].float().unsqueeze(1),isEdge=True)/255. #.squeeze(1).permute(1,2,0).cpu().detach().numpy()

        Le = intensity_loss(edge1, edge2, l_num=2) * 10
        # print(Id.shape,Id.max())
        # print(Le.shape,Le.max())
        Is1 = intensity_loss(seg1[:,:,0:-1,:] , seg1[:,:,1:,:] ,l_num=1) * (Le[:,:,0:-1,:] + Le[:,:,1:,:])
        Is2 = intensity_loss(seg1[:,:,:,0:-1] , seg1[:,:,:,1:] ,l_num=1) * (Le[:,:,:,0:-1] + Le[:,:,:,1:])
        l_s2 = torch.mean(Is1) + torch.mean(Is2)

        l_total = a*l_boundary + b*(l_d + l_s + l_s2)
        # l_total = a*l_boundary + b*(l_d)
        return l_total,a*l_boundary,b*(l_d + l_s + l_s2)


