import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
import argparse
import torch
# from net.dunhuang128_homo_model import HModel
from net.udis512_homo_mesh_model import HModel
from utils.dataSet import HomoOutputData
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"


def inference_test_func(model, optimizer, dataloders):
    print("------------------------------------------")
    print("generating aligned images for testing set")
    model.eval()
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        length = len(dataloders['test'])
        for i, (or_img1,or_img2,re_img1, re_img2, size_tensor) in enumerate(dataloders['test']):
            or_img1 = or_img1.float().to(args.device_ids_alls[0])
            or_img2 = or_img2.float().to(args.device_ids_alls[0])
            re_img1 = re_img1.float().to(args.device_ids_alls[0])
            re_img2 = re_img2.float().to(args.device_ids_alls[0])
            size_tensor = size_tensor.float().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            coarsealignment = model.output_H_estimator(or_img1, or_img2, re_img1, re_img2, size_tensor)

            coarsealignment = coarsealignment[0].cpu().detach().numpy()
            warp1 = coarsealignment[..., 0:3]
            warp2 = coarsealignment[..., 3:6]
            mask1 = coarsealignment[..., 6:9]
            mask2 = coarsealignment[..., 9:12]

            mask1[mask1 > 0.5] = 1
            mask2[mask2 > 0.5] = 1
            mask1[mask1 <= 0.5] = 0
            mask2[mask2 <= 0.5] = 0
            mask1.astype(int)
            mask2.astype(int)
            overlapMask = cv2.bitwise_and(mask1,mask2)
            warp1_m = warp1 * overlapMask
            warp2_m = warp2 * overlapMask

            psnr = compare_psnr(warp1_m, warp2_m , data_range=1)
            ssim = compare_ssim(warp1_m , warp2_m, data_range=1, channel_axis=2)

            if psnr > 100:
                print("input_clip:",or_img1.shape)
                img1 = or_img1[0,...].permute(1,2,0).detach().cpu().numpy()
                img2 = or_img2[0,...].permute(1,2,0).detach().cpu().numpy()
                psnr = compare_psnr(img1, img2 , data_range=1)
                ssim = compare_ssim(img1 , img2, data_range=1, channel_axis=2)

            print('i = {} / {}, psnr = {:.6f}, ssim = {:.6f}'.format(i + 1, length, psnr,ssim))
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    print("===================Results Analysis==================")
    psnr_list.sort(reverse=True)
    psnr_list_30 = psnr_list[0: 514]
    psnr_list_60 = psnr_list[514: 1028]
    psnr_list_100 = psnr_list[1028: -1]
    # psnr_list_30 = psnr_list[0: 11]
    # psnr_list_60 = psnr_list[11: 21]
    # psnr_list_100 = psnr_list[21: -1]
    print("top 30%", np.mean(psnr_list_30))
    print("top 30~60%", np.mean(psnr_list_60))
    print("top 60~100%", np.mean(psnr_list_100))
    print('average psnr:', np.mean(psnr_list))

    ssim_list.sort(reverse=True)
    ssim_list_30 = ssim_list[0: 514]
    ssim_list_60 = ssim_list[514: 1028]
    ssim_list_100 = ssim_list[1028: -1]
    # ssim_list_30 = ssim_list[0: 11]
    # ssim_list_60 = ssim_list[11: 21]
    # ssim_list_100 = ssim_list[21: -1]
    print("top 30%", np.mean(ssim_list_30))
    print("top 30~60%", np.mean(ssim_list_60))
    print("top 60~100%", np.mean(ssim_list_100))
    print('average ssim:', np.mean(ssim_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ############# DunHuang 128 * 128 dataset testing #################
    # parser.add_argument('--path', type=str, default=r'E:\DataSets\DHSD')
    # parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    # parser.add_argument('--img_h', type=int, default=128)
    # parser.add_argument('--img_w', type=int, default=128)
    # parser.add_argument('--output_batch_size', type=int, default=1)
    # parser.add_argument('--max_epoch', type=int, default=200)
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--save_model_name', type=str, default='./dunhuang_checkpoint/homo_model/homo_model_epoch200.pkl')
    ############# UDIS 512 * 512 dataset testing #################
    parser.add_argument('--path', type=str, default=r'D:\learningResource\researchResource\ImageStitiching\dataSet\UDIS-D')
    parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--output_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='./udis_checkpoint/homo_model/UDIS512_homo_mesh_model_epoch150.pkl')
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathTrainInput1 = os.path.join(args.path, 'testing/input1')
    pathTrainInput2 = os.path.join(args.path, 'testing/input2')
    pathTestInput1 = os.path.join(args.path, 'testing/input1')
    pathTestInput2 = os.path.join(args.path, 'testing/input2')

    # test
    image_datasets = {}
    image_datasets['train'] = HomoOutputData(pathTrainInput1, pathTrainInput2, args.img_h, args.img_w)
    image_datasets['test'] = HomoOutputData(pathTestInput1, pathTestInput2, args.img_h, args.img_w)
    dataloders = {}
    dataloders['train'] = DataLoader(image_datasets['train'], batch_size=args.output_batch_size, shuffle=False, num_workers=4)
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.output_batch_size, shuffle=False, num_workers=4)
    # load model
    model = HModel(3)
    pretrain_model=torch.load(args.save_model_name,map_location='cpu')
    # Extract K,V from the existing model
    model_dict = model.state_dict()
    # Create a new weight dictionary and update it
    state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
    # Update the weight dictionary of the existing model
    model_dict.update(state_dict)
    # Load the updated weight dictionary
    model.load_state_dict(model_dict)
    # loading model to device 0
    model = model.to(device=args.device_ids_alls[0])
    model.eval()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    inference_test_func(model, optimizer, dataloders)

