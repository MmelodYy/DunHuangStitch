import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
import argparse
import torch
from thop import profile
# from net.dunhuang128_homo_model import HModel
from net.udis512_homo_mesh_model import HModel
from utils.dataSet import HomoTestData
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def inference_func(model,optimizer,dataloders):
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        length = 1714
        for i, (re_img1, re_img2) in enumerate(dataloders['test']):
            re_img1 = re_img1.float().to(args.device_ids_alls[0])
            re_img2 = re_img2.float().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            _, _, _, _, _, warp, _, _, warp_one = model(
                re_img1, re_img2, re_img1, re_img2)

            warp = warp.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            warp_one = warp_one.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            re_img1 = re_img1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            I1 = re_img1 * warp_one
            I2 = warp * warp_one

            psnr = compare_psnr(I1, I2 , data_range=1)
            ssim = compare_ssim(I1 , I2, data_range=1, channel_axis=2)
            if psnr > 100:
                print("input_clip:", re_img1.shape)
                img1 = re_img1[0, ...].permute(1, 2, 0).detach().cpu().numpy()
                img2 = re_img2[0, ...].permute(1, 2, 0).detach().cpu().numpy()
                psnr = compare_psnr(img1, img2, data_range=1)
                ssim = compare_ssim(img1, img2, data_range=1, channel_axis=2)
            # image fusion
            img1 = re_img1
            img2 = I2
            fusion = np.zeros_like(img1, dtype=np.float64)
            fusion[..., 0] = img2[..., 0]
            fusion[..., 1] = img1[..., 1] * 0.5 + img2[..., 1] * 0.5
            fusion[..., 2] = img1[..., 2]
            fusion = np.clip(fusion, 0, 1)

            # path = "HomoFuse/" + str(i + 1).zfill(5) + ".jpg"
            # print(fusion.shape)
            # cv2.imwrite(path, fusion*255.)
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
    # parser.add_argument('--train_batch_size', type=int, default=8)
    # parser.add_argument('--test_batch_size', type=int, default=1)
    # parser.add_argument('--max_epoch', type=int, default=200)
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--save_model_name', type=str, default='./dunhuang_checkpoint/homo_model/homo_model_epoch200.pkl')
    ############# UDIS 512 * 512 dataset testing #################
    parser.add_argument('--path', type=str, default=r'D:\learningResource\researchResource\ImageStitiching\dataSet\UDIS-D')
    parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='./udis_checkpoint/homo_model/UDIS512_homo_mesh_model_epoch150.pkl')
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathTestInput1 = os.path.join(args.path, 'testing/input1')
    pathTestInput2 = os.path.join(args.path, 'testing/input2')
    # test
    image_datasets = {}
    dataloders = {}
    image_datasets['test'] = HomoTestData(pathTestInput1, pathTestInput2, args.img_h, args.img_w)
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    # load model
    model = HModel(3)
    # model.load_state_dict(torch.load(args.save_model_name))
    # model = model.to(args.gpu_device)

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
    # model.feature_extractor.fuse()
    # model.reg1.fuse()
    # model.reg2.fuse()
    # model.reg3.fuse()
    model.eval()
    ################
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    inputs1_aug = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids_alls[0])
    inputs2_aug = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids_alls[0])
    re_img1 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids_alls[0])
    re_img2 = torch.rand(1, 3, args.img_h, args.img_w).float().to(args.device_ids_alls[0])
    tensor = (inputs1_aug,inputs2_aug,re_img1,re_img2)
    # analysis model params
    macs, params = profile(model, inputs=tensor)
    print("Number of Params: %.2fM" % (params / 1e6))
    print("Number of MACs: %.2fM" % (macs / 1e9))
    ###############
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    # test
    inference_func(model,optimizer,dataloders)

        
        
        


    






