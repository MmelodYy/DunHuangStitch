import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

class CalSeamMetirc_local_global():
    def __init__(self, k_size=5, data_range=255., l_num=2):
        super().__init__()
        self.k_size = k_size
        self.data_range = data_range
        self.l_num = l_num

    def intensity_loss(self, gen_frames, gt_frames, l_num=1):
        return np.mean(np.abs((gen_frames - gt_frames) ** l_num))

    def get_mask(self, mask1, mask2):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.k_size, self.k_size))
        dst1 = cv2.dilate(mask1, kernel)
        dst2 = cv2.dilate(mask2, kernel)
        # dst = dst1 & dst2
        dst = cv2.bitwise_and(dst1, dst2)
        thresh, dst = cv2.threshold(dst, 127.5, 255, cv2.THRESH_BINARY)
        thresh1, dst1 = cv2.threshold(dst1, 127.5, 255, cv2.THRESH_BINARY)
        thresh2, dst2 = cv2.threshold(dst2, 127.5, 255, cv2.THRESH_BINARY)

        dst = dst / 255.
        dst1 = dst1 / 255.
        dst2 = dst2 / 255.
        return dst1, dst2, dst

    def Global_consistent(self, input1, input2, stitchedImg, dst1, dst2):
        img1_m = input1 * dst1
        img2_m = input2 * dst2
        stitchedImg_m1 = stitchedImg * dst1
        stitchedImg_m2 = stitchedImg * dst2
        # calculate
        ssim1 = compare_ssim(img1_m, stitchedImg_m1, data_range=self.data_range, channel_axis=2)
        ssim2 = compare_ssim(img2_m, stitchedImg_m2, data_range=self.data_range, channel_axis=2)
        metirc = (ssim1 + ssim2) / 2.0
        # print(ssim1,ssim2)
        # lp_1 = self.intensity_loss(img1_m, stitchedImg_m1,l_num=self.l_num)
        # lp_2 = self.intensity_loss(img2_m, stitchedImg_m2,l_num=self.l_num)
        # metirc = (lp_1 + lp_2) / 2.0
        # print(lp_1,lp_2)
        return metirc

    # def get_mask(self,mask1,mask2,k_size= 5):
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    #     dst1 = cv2.dilate(mask1, kernel)
    #     dst2 = cv2.dilate(mask2, kernel)
    #     dst = cv2.bitwise_and(dst1,dst2)
    #     thresh, dst = cv2.threshold(dst, 127.5, 255, cv2.THRESH_BINARY)

    #     dst = dst  / 255.
    #     return dst1,dst2,dst

    # def Local_consistent_ssim(self,input1,input2,stitchedImg,dst):
    #     img1_m = input1 * dst
    #     img2_m = input2 * dst
    #     stitchedImg_m = stitchedImg * dst
    #     # calculate
    #     ssim1 = compare_ssim(img1_m, stitchedImg_m, data_range=self.data_range, multichannel=True)
    #     ssim2 = compare_ssim(img2_m, stitchedImg_m, data_range=self.data_range, multichannel=True)
    #     metirc = (2 - ssim1 - ssim2) / 2.0
    #     return metirc

    def Local_consistent(self, input1, input2, stitchedImg, dst):
        img1_m = input1 * dst
        img2_m = input2 * dst
        stitchedImg = stitchedImg * dst
        # calculate
        lp_1 = self.intensity_loss(img1_m, stitchedImg, l_num=self.l_num)
        lp_2 = self.intensity_loss(img2_m, stitchedImg, l_num=self.l_num)
        metirc = (lp_1 + lp_2) / 2.0
        return metirc

    def forward(self, input1, input2, stitchedImg, mask1, mask2):
        # step 1: get mask
        dst1, dst2, dst = self.get_mask(mask1, mask2)
        # step 2: global sturcture consistent
        metirc1 = self.Global_consistent(input1, input2, stitchedImg, dst1, dst2)
        # metirc1 = self.Local_consistent_ssim(input1,input2,stitchedImg,dst)
        # step 3: local seam field consistent
        metirc2 = self.Local_consistent(input1, input2, stitchedImg, dst)
        # step 4: combine global and local
        # print("global ssim:",metirc1," local l2:",metirc2)
        metirc = (metirc1 + metirc2) / 2.0
        return metirc, dst1


class CalSeamMetirc():
    def __init__(self, k_size_lp=5, k_size_ssim=5, data_range=255., l_num=2):
        super().__init__()
        self.k_size_ssim = k_size_ssim
        self.k_size_lp = k_size_lp
        self.data_range = data_range
        self.l_num = l_num

    def intensity_loss(self, gen_frames, gt_frames, l_num=1):
        return np.mean(np.abs((gen_frames - gt_frames) ** l_num))

    def get_mask(self, mask1, mask2, k_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dst1 = cv2.dilate(mask1, kernel)
        dst2 = cv2.dilate(mask2, kernel)
        dst = cv2.bitwise_and(dst1, dst2)
        thresh, dst = cv2.threshold(dst, 127.5, 255, cv2.THRESH_BINARY)

        dst = dst / 255.
        return dst

    def Local_consistent_ssim(self, input1, input2, stitchedImg, dst):
        img1_m = input1 * dst
        img2_m = input2 * dst
        stitchedImg_m = stitchedImg * dst
        # calculate
        ssim1 = compare_ssim(img1_m, stitchedImg_m, data_range=self.data_range, channel_axis=2)
        ssim2 = compare_ssim(img2_m, stitchedImg_m, data_range=self.data_range, channel_axis=2)
        # print("ssim1:",ssim1," ssim2:",ssim2)
        metirc = (2 - ssim1 - ssim2) / 2.0
        return metirc

    def Local_consistent(self, input1, input2, stitchedImg, dst):
        img1_m = input1 * dst
        img2_m = input2 * dst
        stitchedImg = stitchedImg * dst
        # calculate
        lp_1 = self.intensity_loss(img1_m, stitchedImg, l_num=self.l_num)
        lp_2 = self.intensity_loss(img2_m, stitchedImg, l_num=self.l_num)
        metirc = (lp_1 + lp_2) / 2.0
        return metirc

    # def forward(self,input1,input2,stitchedImg,mask1,mask2):
    #     # step 1: get ssim mask
    #     dst1 = self.get_mask(mask1,mask2,self.k_size_ssim)
    #     # step 2: global sturcture consistent
    #     metirc1 = self.Local_consistent_ssim(input1,input2,stitchedImg,dst1)
    #     # step 3: get mask
    #     dst2 = self.get_mask(mask1,mask2,self.k_size_lp)
    #     # step 4: local seam field consistent
    #     metirc2 = self.Local_consistent(input1,input2,stitchedImg,dst2)
    #     # step 5: combine global and local
    #     # print("global ssim:",metirc1," local l2:",metirc2)
    #     metirc = metirc2# (metirc1 + metirc2)/2.0
    #     return metirc,dst1

    def forward(self, input1, input2, stitchedImg, mask1, mask2):
        # step 1: get ssim mask
        dst = self.get_mask(mask1, mask2, self.k_size_ssim)
        # step 2: global sturcture consistent
        metirc1 = self.Local_consistent_ssim(input1, input2, stitchedImg, dst)
        # step 4: local seam field consistent
        # print("input1:",input1.max()," stitchedImg:",stitchedImg.max()," dst:",dst.max())
        metirc2 = self.Local_consistent(input1, input2, stitchedImg, dst)
        # step 5: combine global and local
        # print("global ssim:",metirc1*1000," local l2:",metirc2)
        # metirc = (metirc1 + metirc2)/2.0
        # alpha = 500
        # metirc = sigmoid(metirc1 * metirc2 * alpha)
        # metirc = (metirc - 0.5)/(1 - 0.5)
        ##############
        metirc = metirc1 * 1000 + metirc2
        return metirc, dst



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    path = './dunhuang_output/stitch_output'
    path_or = './dunhuang_output/homo_output/testing'

    or_img_list = list(sorted(os.listdir(os.path.join(path_or, 'warp1'))))
    sq_list = []

    data_range = 255
    k_size = 5
    l_num = 2
    # metirc_model = CalSeamMetirc_local_global(k_size=k_size,data_range=data_range,l_num=l_num)
    metirc_model = CalSeamMetirc(k_size_lp=k_size, data_range=data_range, l_num=l_num)
    for i in range(len(or_img_list)):
        mask1_path = os.path.join(path, 'learn_mask1/' + or_img_list[i])
        mask2_path = os.path.join(path, 'learn_mask2/' + or_img_list[i])
        composition_img_path = os.path.join(path, 'composition/' + or_img_list[i])
        img1_path = os.path.join(path_or, 'warp1/' + or_img_list[i])
        img2_path = os.path.join(path_or, 'warp2/' + or_img_list[i])

        mask1 = cv2.imread(mask1_path)
        mask2 = cv2.imread(mask2_path)
        composition_img = cv2.imread(composition_img_path)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        # print(img1.shape, mask1.shape)

        sq, dst1 = metirc_model.forward(img1, img2, composition_img, mask1, mask2)
        # img2_path = os.path.join(path, 'demo/' + or_img_list[i])
        # cv2.imwrite(img2_path, dst1 * 255.)

        print("i = {}, image name = {},seam quality = {:.4f}".format(i + 1, or_img_list[i], sq))
        sq_list.append(sq)
    print("===================Results Analysis==================")
    sq_list.sort(reverse=True)
    sq_list_30 = sq_list[0: 514]
    sq_list_60 = sq_list[514: 1028]
    sq_list_100 = sq_list[1028: -1]
    print("top 30%", np.mean(sq_list_30))
    print("top 30~60%", np.mean(sq_list_60))
    print("top 60~100%", np.mean(sq_list_100))
    print('average seam quality:', np.mean(sq_list))









