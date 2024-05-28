import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
import argparse
import torch
from torch import nn
from net.fast_stitching_model import FastStitch,reparameterize_model
from utils.dataSet import FuseData
from torch.utils.data import DataLoader
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def inference_func(model,optimizer,dataloders):
    with torch.no_grad():
        length = 1714
        for i, (or_img1,or_img2,  mask1,mask2) in enumerate(dataloders['test']):
            optimizer.zero_grad()
            or_img1 = or_img1.float().to(args.device_ids_alls[0])
            or_img2 = or_img2.float().to(args.device_ids_alls[0])
            mask1 = mask1.int().to(args.device_ids_alls[0])
            mask2 = mask2.int().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            stitched_img, seg1, seg2 = model(or_img1, or_img2,  mask1, mask2)

            stitched_img = stitched_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.
            seg1 = seg1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.
            seg2 = seg2.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.
            or_img1 = or_img1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.
            or_img2 = or_img2.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.

            s1 = or_img1 * (seg1 / 255.)
            s2 = or_img2 *(seg2 / 255.)

            fusion = np.zeros_like(s1)
            fusion[...,0] = s2[...,0]
            fusion[...,1] = s1[...,1]*0.5 +  s2[...,1]*0.5
            fusion[...,2] = s1[...,2]
            path_fusion = args.base_path + "/fusion_color/" + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path_fusion, fusion)

            path = args.base_path + "/composition/" + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path, stitched_img)
            path1 = args.base_path + "/learn_mask1/" + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path1, seg1)
            path2 = args.base_path + "/learn_mask2/" + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path2, seg2)

            print('i = {} / {}'.format(i + 1, length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ############# DunHuang 128 * 128 dataset testing #################
    # parser.add_argument('--path', type=str, default='./dunhuang_output/homo_output')
    # parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    # parser.add_argument('--img_h', type=int, default=128)
    # parser.add_argument('--img_w', type=int, default=128)
    # parser.add_argument('--train_batch_size', type=int, default=8)
    # parser.add_argument('--test_batch_size', type=int, default=1)
    # parser.add_argument('--max_epoch', type=int, default=200)
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--base_path', type=str, default='./dunhuang_output/stitch_output')
    # parser.add_argument('--save_model_name', type=str, default='dunhuang_checkpoint/stitch_model/stitch_model_epoch50.pkl')
    ############# DUIS 512 * 512 dataset testing #################
    parser.add_argument('--path', type=str, default='./udis_output/homo_output')
    parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--base_path', type=str, default='./udis_output/stitch_output')
    parser.add_argument('--save_model_name', type=str, default='udis_checkpoint/stitch_model/udis512_stitch_model_epoch50.pkl')
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathTestInput1 = os.path.join(args.path, 'testing/warp1')
    pathTestInput2 = os.path.join(args.path, 'testing/warp2')
    pathTestMask1 = os.path.join(args.path, 'testing/mask1')
    pathTestMask2 = os.path.join(args.path, 'testing/mask2')
    # test
    image_datasets = {}
    dataloders = {}
    image_datasets = {}
    image_datasets['test'] = FuseData(pathTestInput1, pathTestInput2, pathTestMask1, pathTestMask2, args.img_h, args.img_w)
    dataloders = {}
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    # loading model
    params = {
        "in_chans_list": (32, 32, 48, 64, 96),
        "out_chans_list": (32, 48, 64, 96, 128),
        "blocks_per_stage": (2, 2, 2, 2, 2),
        "expand_ratio": (2, 2, 2, 2, 2),
        "use_attn": (False, False, False, False, True),
        "use_patchEmb": (False, True, True, True, True),
    }
    model = FastStitch(inchannels=3, inference_mode=False,
                       act_layer=nn.GELU, **params).to(args.device_ids_alls[0])
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
    model = reparameterize_model(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    # test
    inference_func(model,optimizer,dataloders)

        
        
        


    






