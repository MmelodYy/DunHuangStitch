import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
import cv2
import torch
from torch import nn
import random
import argparse
from tqdm import tqdm
from utils.dataSet import FuseData
from utils.learningRateScheduler import warmUpLearningRate
from net.loss_functions import FuseLoss
from net.fast_stitching_model import FastStitch
import numpy as np
from thop import profile
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

setup_seed(114514)

def train_once(model,each_data_batch,epoch,epochs,criterion, optimizer):
    model.train()
    with tqdm(total=each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        each_batch_all_loss = 0
        each_batch_lp_loss = 0
        each_batch_ssim_loss = 0
        print("Start Train")
        for i, (or_img1,or_img2, mask1,mask2) in enumerate(dataloders['train']):
            or_img1 = or_img1.float().to(args.device_ids_alls[0])
            or_img2 = or_img2.float().to(args.device_ids_alls[0])
            mask1 = mask1.int().to(args.device_ids_alls[0])
            mask2 = mask2.int().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            stitched_img,seg1,seg2 = model(or_img1,or_img2,mask1,mask2)
            loss,lp,ssim = criterion(or_img1,or_img2,mask1,mask2,stitched_img,seg1,seg2)
            loss.backward()
            optimizer.step()

            each_batch_all_loss += loss.item() / args.train_batch_size
            each_batch_lp_loss += lp.item() / args.train_batch_size
            each_batch_ssim_loss += ssim.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Llp': each_batch_lp_loss / (i + 1),
                              'Lssim': each_batch_ssim_loss / (i + 1),
                              'lr': scheduler.get_last_lr()[0]})
            pbar.update(1)
        print("\nFinish Train")
        return each_batch_all_loss / each_data_batch

def train(model,saveModelName,criterion,optimizer,scheduler,epochs=1):
    loss_history = []
    each_data_batch = len(dataloders['train'])
    for epoch in range(epochs):
        # 训练
        each_batch_all_loss = train_once(model,each_data_batch,epoch, epochs,criterion,optimizer)

        # learning rate scheduler
        scheduler.step()
        # print("epoch:",epoch,"lr:",scheduler.get_last_lr())
        loss_history.append(each_batch_all_loss)

        if epoch % 10 == 0 or (epoch + 1) >= (epochs - 10):
            torch.save(model.state_dict(), saveModelName + "_" + "epoch" + str(epoch + 1) + ".pkl")
            np.save(saveModelName + "_" + "epoch" + str(epoch + 1) + "_" + "TrainLoss" +
            str(round(each_batch_all_loss, 3)), np.array(loss_history))

    show_plot(loss_history)

def show_plot(loss_history):
    counter = range(len(loss_history))
    plt.plot(counter, loss_history)
    plt.legend(['train loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ############# DunHuang 128 * 128 dataset training #################
    # parser.add_argument('--path', type=str, default='./dunhuang_output/homo_output')
    # parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    # parser.add_argument('--img_h', type=int, default=128)
    # parser.add_argument('--img_w', type=int, default=128)
    # parser.add_argument('--train_batch_size', type=int, default=1)
    # parser.add_argument('--test_batch_size', type=int, default=1)
    # parser.add_argument('--max_epoch', type=int, default=50)
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--save_model_name', type=str, default='./dunhuang_checkpoint/stitch_model/stitch_model')
    ############# DUIS 512 * 512 dataset training #################
    parser.add_argument('--path', type=str, default='./udis_output/homo_output')
    parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='./udis_checkpoint/stitch_model/udis512_stitch_model')
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathTrainInput1 = os.path.join(args.path, 'training/warp1')
    pathTrainInput2 = os.path.join(args.path, 'training/warp2')
    pathTrainMask1 = os.path.join(args.path, 'training/mask1')
    pathTrainMask2 = os.path.join(args.path, 'training/mask2')

    image_datasets = {}
    image_datasets['train'] = FuseData(pathTrainInput1, pathTrainInput2, pathTrainMask1, pathTrainMask2, args.img_h, args.img_w)
    dataloders = {}
    dataloders['train'] = DataLoader(image_datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    # define somethings
    criterion = FuseLoss().to(args.device_ids_alls[0])
    params = {
            "in_chans_list": (32, 32, 48, 64, 96),
            "out_chans_list": (32, 48, 64, 96, 128),
            "blocks_per_stage": (2, 2, 2, 2, 2),
            "expand_ratio": (2, 2, 2, 2, 2),
            "use_attn": (False, False, False, False, True),
            "use_patchEmb": (False, True, True, True, True),
        }
    model = FastStitch(inchannels=3, inference_mode= False,
                                                     act_layer=nn.GELU, **params).to(args.device_ids_alls[0])

    ################
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    or_img1 = torch.rand(1, 3, args.img_h*2, args.img_w*2).float().to(args.device_ids_alls[0])
    or_img2 = torch.rand(1, 3, args.img_h*2, args.img_w*2).float().to(args.device_ids_alls[0])
    mask1 = torch.rand(1, 3, args.img_h*2, args.img_w*2).int().to(args.device_ids_alls[0])
    mask2 = torch.rand(1, 3, args.img_h*2, args.img_w*2).int().to(args.device_ids_alls[0])
    tensor = (or_img1, or_img2, mask1,mask2)
    # analysis model params
    macs, params = profile(model, inputs=tensor)
    print("Number of Params: %.2f M" % (params / 1e6))
    print("Number of MACs: %.2f G" % (macs / 1e9))
    ###############
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
    lrScheduler = warmUpLearningRate(args.max_epoch, warm_up_epochs=5, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrScheduler)
    train(model, args.save_model_name, criterion, optimizer, scheduler, args.max_epoch)