import argparse
import torch
import os
from tqdm import tqdm
from utils.dataSet import HomoTrainData,HomoTestData
from utils.learningRateScheduler import warmUpLearningRate
from net.loss_functions import Homo_Mesh_Total_Loss
from net.udis512_homo_mesh_model import HModel
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
import random
from thop import profile
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

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
        each_batch_mesh_loss = 0
        print("Start Train")
        for i, (inputs1_aug,inputs2_aug,re_img1,re_img2) in enumerate(dataloders['train']):
            inputs1_aug = inputs1_aug.float().to(args.device_ids_alls[0])
            inputs2_aug = inputs2_aug.float().to(args.device_ids_alls[0])
            re_img1 = re_img1.float().to(args.device_ids_alls[0])
            re_img2 = re_img2.float().to(args.device_ids_alls[0])

            optimizer.zero_grad()
            # train_net1_f, train_net2_f, train_net3_f, \
            # train_warp2_H1, train_warp2_H2, train_warp2_H3, \
            # train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = model(
            #     inputs1_aug, inputs2_aug, re_img1, re_img2)
            # loss = criterion(re_img1,train_warp2_H1, train_warp2_H2, train_warp2_H3, \
            # train_one_warp_H1, train_one_warp_H2, train_one_warp_H3)
            ################ Mesh Network ###################
            train_net1_f, train_net2_f, train_mesh, \
                train_warp2_H1, train_warp2_H2, train_warp2_H3, \
                train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = model(
                inputs1_aug, inputs2_aug, re_img1, re_img2)
            loss, loss_lp, mesh_loss = criterion(train_mesh, re_img1, train_warp2_H1, train_warp2_H2, train_warp2_H3, \
                                                 train_one_warp_H1, train_one_warp_H2, train_one_warp_H3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()

            each_batch_all_loss += loss.item() / args.train_batch_size
            each_batch_lp_loss += loss_lp.item() / args.train_batch_size
            each_batch_mesh_loss += mesh_loss.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Lp': each_batch_lp_loss / (i + 1),
                              'Lmesh': each_batch_mesh_loss / (i + 1),
                              'lr': scheduler.get_last_lr()[0]})
            pbar.update(1)
        print("\nFinish Train")
        return each_batch_all_loss / each_data_batch

def val_once(model,t_each_data_batch,epoch,epochs,criterion,optimizer):
    model.eval()
    with tqdm(total=t_each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as t_pbar:
        each_batch_psnr = 0
        each_batch_ssim = 0
        each_batch_all_loss = 0
        print("Start Test")
        with torch.no_grad():
            for i, (re_img1,re_img2) in enumerate(dataloders['test']):
                re_img1 = re_img1.float().to(args.device_ids_alls[0])
                re_img2 = re_img2.float().to(args.device_ids_alls[0])

                optimizer.zero_grad()
                # net1_f, net2_f, net3_f,  warp2_H1, warp2_H2, warp2_H3, \
                # one_warp_H1, one_warp_H2, one_warp_H3= model(
                #     re_img1, re_img2, re_img1, re_img2)
                #
                # loss = criterion(re_img1, warp2_H1, warp2_H2, warp2_H3, \
                #                  one_warp_H1, one_warp_H2, one_warp_H3)
                ############ Mesh Network ############
                net1_f, net2_f, mesh, \
                    warp2_H1, warp2_H2, warp2_H3, \
                    one_warp_H1, one_warp_H2, one_warp_H3 = model(
                    re_img1, re_img2, re_img1, re_img2)
                loss, loss_lp, mesh_loss = criterion(mesh, re_img1, warp2_H1, warp2_H2,
                                                     warp2_H3, \
                                                     one_warp_H1, one_warp_H2, one_warp_H3)

                warp = warp2_H3.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                warp_one = one_warp_H3.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                re_img1 = re_img1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                I1 = re_img1*warp_one
                I2 = warp*warp_one

                psnr = compare_psnr(I1, I2 , data_range=1)
                ssim = compare_ssim(I1 , I2, data_range=1, channel_axis=2)

                each_batch_psnr += psnr / args.test_batch_size
                each_batch_ssim += ssim / args.test_batch_size
                each_batch_all_loss += loss.item() / args.test_batch_size
                t_pbar.set_postfix({'Test Loss': each_batch_all_loss / (i + 1),
                                    'average psnr': each_batch_psnr / (i + 1),
                                    'average ssim': each_batch_ssim / (i + 1)})
                t_pbar.update(1)
        print("\nFinish Test")

def train(model,saveModelName,criterion,optimizer,scheduler,start_epochs=0,end_epochs=1):
    loss_history = []
    each_data_batch = len(dataloders['train'])
    t_each_data_batch = len(dataloders['test'])
    for epoch in range(start_epochs,end_epochs):
        # 训练
        each_batch_all_loss = train_once(model,each_data_batch,epoch, end_epochs,criterion,optimizer)
        if epoch % 10 == 0:
            # 测试
            val_once(model, t_each_data_batch, epoch, end_epochs, criterion, optimizer)
        # learning rate scheduler
        scheduler.step()
        # print("epoch:",epoch,"lr:",scheduler.get_last_lr())
        loss_history.append(each_batch_all_loss)

        # Meiyuan is hadnsome!
        if epoch % 10 == 0 or (epoch + 1) >= (end_epochs - 10):
            save_path = saveModelName.split('/')[0] + '/'
            if not os.path.exists(save_path):
                print("---create model dir---")
                os.mkdir(save_path)
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
    ############# UDIS 512 * 512 dataset training #################
    ############# 1. homo + homo + homo #################

    ############# 2. homo + homo + mesh #################
    parser.add_argument('--path', type=str, default=r'D:\learningResource\researchResource\ImageStitiching\dataSet\UDIS-D')
    parser.add_argument('--device_ids_alls', type=list, default=[0], help='Number of splits')
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--start_epochs', type=int, default=0)
    # parser.add_argument('--end_epochs', type=int, default=200)
    parser.add_argument('--end_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='udis_checkpoint/homo_model/UDIS512_homo_mesh_model')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_epoch', type=int, default=121)
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathTrainInput1 = os.path.join(args.path, 'training/input1')
    pathTrainInput2 = os.path.join(args.path, 'training/input2')
    pathTestInput1 = os.path.join(args.path, 'testing/input1')
    pathTestInput2 = os.path.join(args.path, 'testing/input2')
    image_datasets = {}
    image_datasets['train'] = HomoTrainData(pathTrainInput1,pathTrainInput2, args.img_h, args.img_w)
    image_datasets['test'] = HomoTestData(pathTestInput1,pathTestInput2, args.img_h, args.img_w)
    dataloders = {}
    # print("data:",next(iter(image_datasets['train'])))
    dataloders['train'] = DataLoader(image_datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    # define somethings
    criterion = Homo_Mesh_Total_Loss(l_num=1).to(args.device_ids_alls[0])

    model = HModel(3).to(args.device_ids_alls[0])

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
    print("Number of Params: %.2f M" % (params / 1e6))
    print("Number of MACs: %.2f G" % (macs / 1e9))
    ###############
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
    lrScheduler = warmUpLearningRate(args.end_epochs, warm_up_epochs=20, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrScheduler)
    # resume training
    if args.resume:
        for i in range(0, (args.start_epochs + 1)):
            scheduler.step()
        args.start_epochs = args.resume_epoch
        load_path = args.save_model_name + "_" + "epoch" + str(args.resume_epoch) + ".pkl"
        params = torch.load(load_path)
        model.load_state_dict(params)
        print("-----resume train, load model state dict success------")
    train(model, args.save_model_name, criterion, optimizer, scheduler,  args.start_epochs, args.end_epochs)
