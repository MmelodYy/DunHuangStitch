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

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"


def inference_train_func(model, optimizer, dataloders):
    print("------------------------------------------")
    print("generating aligned images for training set")
    model.eval()
    with torch.no_grad():
        length = 8051
        for i, (or_img1,or_img2,re_img1, re_img2, size_tensor) in enumerate(dataloders['train']):
            or_img1 = or_img1.float().to(args.device_ids_alls[0])
            or_img2 = or_img2.float().to(args.device_ids_alls[0])
            re_img1 = re_img1.float().to(args.device_ids_alls[0])
            re_img2 = re_img2.float().to(args.device_ids_alls[0])
            size_tensor = size_tensor.float().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            coarsealignment = model.output_H_estimator(or_img1, or_img2, re_img1, re_img2, size_tensor)

            coarsealignment = coarsealignment[0].cpu().detach().numpy()
            warp1 = coarsealignment[..., 0:3] * 255.
            warp2 = coarsealignment[..., 3:6] * 255.
            mask1 = coarsealignment[..., 6:9] * 255.
            mask2 = coarsealignment[..., 9:12] * 255.

            path1 = args.base_path + '/training/warp1/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path1, warp1)
            path2 = args.base_path + '/training/warp2/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path2, warp2)
            path3 = args.base_path + '/training/mask1/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path3, mask1)
            path4 = args.base_path + '/training/mask2/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path4, mask2)
            print('i = {} / {}'.format(i + 1, length))

    print("-----------training set done--------------")
    print("------------------------------------------")


def inference_test_func(model, optimizer, dataloders):
    print("------------------------------------------")
    print("generating aligned images for testing set")
    model.eval()
    with torch.no_grad():
        length = 1714
        for i, (or_img1,or_img2,re_img1, re_img2, size_tensor) in enumerate(dataloders['test']):
            or_img1 = or_img1.float().to(args.device_ids_alls[0])
            or_img2 = or_img2.float().to(args.device_ids_alls[0])
            re_img1 = re_img1.float().to(args.device_ids_alls[0])
            re_img2 = re_img2.float().to(args.device_ids_alls[0])
            size_tensor = size_tensor.float().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            coarsealignment = model.output_H_estimator(or_img1, or_img2, re_img1, re_img2, size_tensor)

            coarsealignment = coarsealignment[0].cpu().detach().numpy()
            warp1 = coarsealignment[..., 0:3] * 255.
            warp2 = coarsealignment[..., 3:6] * 255.
            mask1 = coarsealignment[..., 6:9] * 255.
            mask2 = coarsealignment[..., 9:12] * 255.

            path1 = args.base_path + '/testing/warp1/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path1, warp1)
            path2 = args.base_path + '/testing/warp2/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path2, warp2)
            path3 = args.base_path + '/testing/mask1/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path3, mask1)
            path4 = args.base_path + '/testing/mask2/' + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path4, mask2)
            print('i = {} / {}'.format(i + 1, length))

    print("-----------testing set done--------------")
    print("------------------------------------------")



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
    # parser.add_argument('--base_path', type=str, default='./dunhuang_output/homo_output')
    # parser.add_argument('--save_model_name', type=str, default='./dunhuang_checkpoint/homo_model/homo_model_epoch200.pkl')
    ############# UDIS 512 * 512 dataset testing #################
    parser.add_argument('--path', type=str, default=r'D:\learningResource\researchResource\ImageStitiching\dataSet\UDIS-D')
    parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--output_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--base_path', type=str, default='./udis_output/homo_output')
    parser.add_argument('--save_model_name', type=str, default='./udis_checkpoint/homo_model/UDIS512_homo_mesh_model_epoch150.pkl')
    #############################################################

    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathTrainInput1 = os.path.join(args.path, 'training/input1')
    pathTrainInput2 = os.path.join(args.path, 'training/input2')
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
    model_dict=model.state_dict()
    # Create a new weight dictionary and update it
    state_dict={k:v for k,v in pretrain_model.items() if k in model_dict.keys()}
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
    # generate train data
    # inference_train_func(model, optimizer, dataloders)
    # generate test data
    inference_test_func(model, optimizer, dataloders)

