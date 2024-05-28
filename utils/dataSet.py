import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)

# def disjoint_augment_image_pair(img1,img2, min_val=0, max_val=255.):
#     # Randomly shift brightness
#     random_brightness = np.random.uniform(0.7, 1.3)
#     img1_aug = img1 * random_brightness
#     random_brightness = np.random.uniform( 0.7, 1.3)
#     img2_aug = img2 * random_brightness
#
#     # Randomly shift color
#     random_colors = np.random.uniform(0.7, 1.3,size=3)
#     white = np.ones([np.shape(img1)[0], np.shape(img1)[1], np.shape(img1)[2]])
#     # print(white.shape)
#     color_image = white * random_colors
#     # print(color_image.shape,img2_aug.shape)
#     img1_aug *= color_image
#
#     random_colors = np.random.uniform(0.7, 1.3,size=3)
#     # white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1], tf.shape(img1)[2]])
#     color_image =  white * random_colors
#     img2_aug *= color_image
#
#     # Saturate
#     img1_aug = np.uint8(np.clip(img1_aug, min_val, max_val))
#     img2_aug = np.uint8(np.clip(img2_aug, min_val, max_val))
#
#     return img1_aug, img2_aug

# import math
# def random_perspective(image,degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
#     # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#     # targets = [cls, xyxy]
#
#     height = image.shape[0] + border[0] * 2  # shape(h,w,c)
#     width = image.shape[1] + border[1] * 2
#
#     # Center
#     C = np.eye(3)
#     C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
#     C[1, 2] = -image.shape[0] / 2  # y translation (pixels)
#
#     # Perspective
#     P = np.eye(3)
#     P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
#     P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
#
#     # Rotation and Scale
#     R = np.eye(3)
#     a = random.uniform(-degrees, degrees)
#     # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
#     s = random.uniform(1 - scale, 1 + scale)
#     # s = 2 ** random.uniform(-scale, scale)
#     R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
#
#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
#
#     # Translation
#     T = np.eye(3)
#     T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
#     T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
#
#     # Combined rotation matrix
#     M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
#     if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
#         if perspective:
#             image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(0, 0, 0))
#         else:  # affine
#             image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
#
#     return image


class HomoTrainData(nn.Module):
    def __init__(self, data_folder1,data_folder2,img_h,img_w):
        super(HomoTrainData, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(data_folder1)]))
        self.data_folder1 = data_folder1
        self.data_folder2 = data_folder2
        self.img_h = img_h
        self.img_w = img_w

        self._origin_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self._origin_transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=[0.01,0.05]),
            transforms.ColorJitter(contrast=[0.3,0.6]),
            transforms.ColorJitter(saturation=[0.2,0.5]),
            transforms.ColorJitter(hue=[-0.1,0.2]),
            transforms.Resize([self.img_h, self.img_w]),
            transforms.ToTensor(),
        ])
        setup_seed(114514)

    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]
        re_img1 =  cv2.imread(os.path.join(self.data_folder1, idx + '.jpg'))
        re_img2 = cv2.imread(os.path.join(self.data_folder2, idx + '.jpg'))

        re_img1 = Image.fromarray(re_img1)
        re_img2 = Image.fromarray(re_img2)
        inputs1_aug = re_img1.copy()
        inputs2_aug = re_img2.copy()

        inputs1_aug = self._origin_transform_aug(inputs1_aug)
        inputs2_aug = self._origin_transform_aug(inputs2_aug)
        re_img1 = self._origin_transform(re_img1)
        re_img2 = self._origin_transform(re_img2)
        return inputs1_aug, inputs2_aug, re_img1, re_img2


class HomoTestData(nn.Module):
    def __init__(self, data_folder1,data_folder2,img_h,img_w):
        super(HomoTestData, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(data_folder1)]))
        self.data_folder1 = data_folder1
        self.data_folder2 = data_folder2
        self.img_h = img_h
        self.img_w = img_w

        self._origin_transform_re = transforms.Compose([
            transforms.Resize([self.img_h, self.img_w]),
            transforms.ToTensor(),
        ])
        setup_seed(114514)

    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]
        or_img1 =  cv2.imread(os.path.join(self.data_folder1, idx + '.jpg'))
        or_img2 = cv2.imread(os.path.join(self.data_folder2, idx + '.jpg'))

        or_img1 = Image.fromarray(or_img1)
        or_img2 = Image.fromarray(or_img2)

        re_img1 = self._origin_transform_re(or_img1)
        re_img2 = self._origin_transform_re(or_img2)
        return re_img1, re_img2


class HomoOutputData(nn.Module):
    def __init__(self, data_folder1,data_folder2,img_h,img_w):
        super(HomoOutputData, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(data_folder1)]))
        self.data_folder1 = data_folder1
        self.data_folder2 = data_folder2
        self.img_h = img_h
        self.img_w = img_w

        self._origin_transform_re = transforms.Compose([
            transforms.Resize([self.img_h, self.img_w]),
            transforms.ToTensor(),
        ])
        self._origin_transform = transforms.Compose([
            transforms.Resize([self.img_h, self.img_w]),
            transforms.ToTensor(),
        ])
        setup_seed(114514)

    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]
        or_img1 =  cv2.imread(os.path.join(self.data_folder1, idx + '.jpg'))
        or_img2 = cv2.imread(os.path.join(self.data_folder2, idx + '.jpg'))
        # print(or_img2.shape)
        height, width = or_img2.shape[:2]
        # size_tensor = torch.tensor([width, height])
        # resize 128
        size_tensor = torch.tensor([self.img_h, self.img_w])
        # print("size_tensor:", size_tensor.shape)
        or_img1 = Image.fromarray(or_img1)
        or_img2 = Image.fromarray(or_img2)

        re_img1 = self._origin_transform_re(or_img1.copy())
        re_img2 = self._origin_transform_re(or_img2.copy())
        or_img1 = self._origin_transform(or_img1)
        or_img2 = self._origin_transform(or_img2)
        return or_img1, or_img2, re_img1, re_img2, size_tensor

    
class FuseData(nn.Module):
    def __init__(self, data_folder1,data_folder2,data_mask_folder1,data_mask_folder2,img_h,img_w):
        super(FuseData, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(data_folder1)]))
        self.data_folder1 = data_folder1
        self.data_folder2 = data_folder2
        self.data_mask_folder1 = data_mask_folder1
        self.data_mask_folder2 = data_mask_folder2
        self.img_h = img_h
        self.img_w = img_w

        self._origin_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self._origin_transform_resize = transforms.Compose([
            transforms.Resize([self.img_h, self.img_w]),
            transforms.ToTensor(),
        ])
        setup_seed(114514)

    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]

        or_img1 =  cv2.imread(os.path.join(self.data_folder1, idx + '.jpg'))
        or_img2 = cv2.imread(os.path.join(self.data_folder2, idx + '.jpg'))
        mask1 = cv2.imread(os.path.join(self.data_mask_folder1, idx + '.jpg'))
        mask2 = cv2.imread(os.path.join(self.data_mask_folder2, idx + '.jpg'))

        mask1[mask1 < 128] = 0
        mask1[mask1 >= 128] = 255
        mask2[mask2 < 128] = 0
        mask2[mask2 >= 128] = 255

        or_img1 = Image.fromarray(or_img1)
        or_img2 = Image.fromarray(or_img2)
        mask1 = Image.fromarray(mask1)
        mask2 = Image.fromarray(mask2)

        or_img1 = self._origin_transform(or_img1)
        or_img2 = self._origin_transform(or_img2)
        mask1 = self._origin_transform(mask1)
        mask2 = self._origin_transform(mask2)
        return or_img1, or_img2, mask1, mask2
