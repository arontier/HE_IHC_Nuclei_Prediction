# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import PIL
from PIL import Image
import random
from imgaug import augmenters as iaa
from Augmentation import RandomRotate, ImgGeoTransform, ImgGaussBlur, Color_Deconvolution
from Class_ID_Name import CLASS_ID_NAME_Nuclei_Seg
from Nuclei_Seg_Data_Loader_Border import get_loader
from Unet_1 import UNet_1
from Unet_2 import UNet_2
from collections import defaultdict
import torch.nn.functional as F
from Evaluation_Metrics import *

import warnings
warnings.filterwarnings("ignore")

def calc_dice_loss(pred, target,):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def calc_loss(pred, target, bce_weight=0.5):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice_loss = calc_dice_loss(pred, target)
    total_loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)
    return bce_loss, dice_loss, total_loss

def calc_loss_border(pred, target, border_weight=0.75):
    bce_loss_nuclei = F.binary_cross_entropy_with_logits(pred[:,0,:,:], target[:,0,:,:])
    bce_loss_border = F.binary_cross_entropy_with_logits(pred[:,1,:,:], target[:,1,:,:])

    pred = F.sigmoid(pred)
    dice_loss_nuclei = calc_dice_loss(pred[:,0,:,:], target[:,0,:,:])
    dice_loss_border = calc_dice_loss(pred[:,1,:,:], target[:,1,:,:])

    nuclei_loss = bce_loss_nuclei + dice_loss_nuclei
    border_loss = bce_loss_border + dice_loss_border

    total_loss = nuclei_loss * (1 - border_weight) + border_loss * border_weight
    return nuclei_loss, border_loss, total_loss




def train_model(model,dataloaders,dataset_sizes, optimizer, scheduler,Weight_Save_Dir, device="cuda:0", epoch_load=1, num_epochs=100):
    since = time.time()

    train_logs = {'train_loss':[],
                  'train_dc': [],
                  'val_loss':[],
                  'val_dc': [],
                  # 'val_f1': [],
                  # 'val_aji': [],
                  'lr':[]}
    best_DC=0
    batch_iter = 1
    for epoch in range(epoch_load, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_F1 = 0.0
            running_AJI = 0.0
            running_Nuclei_DC = 0.0
            running_Border_DC = 0.0

            epoch_batch_counting = 1
                # Iterate over data.
            for inputs, masks, inputs_ori, img_name in dataloaders[phase]:

                inputs = inputs.to(device)
                masks = masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    outputs_sigmoid = F.sigmoid(outputs)
                    # bce_loss, dice_loss, total_loss = calc_loss(outputs, masks)
                    nuclei_loss, border_loss, total_loss = calc_loss_border(outputs, masks, border_weight=0.75)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        total_loss.backward()
                        scheduler.step()
                        optimizer.step()
                        for param_group in optimizer.param_groups:
                            tmp_lr = param_group['lr']
                        # print('Batch Iteration: {},  Training Steps: {}/{},  BCE: {:.4f}, DICE: {:.4f}, Loss: {:.4f},  Lr: {}'.format(batch_iter, epoch_batch_counting*inputs.size()[0],dataset_sizes[phase], bce_loss, dice_loss, total_loss, tmp_lr))
                        print('Batch Iteration: {},  Training Steps: {}/{},  Nuclei Loss: {:.4f}, Border Loss: {:.4f}, Total Loss: {:.4f},  Lr: {}'.format(batch_iter, epoch_batch_counting*inputs.size()[0],dataset_sizes[phase], nuclei_loss, border_loss, total_loss, tmp_lr))

                        batch_iter = batch_iter+1
                        epoch_batch_counting = epoch_batch_counting +1

                    else:
                        print('Validation Steps: {}/{}'.format(epoch_batch_counting*inputs.size()[0],dataset_sizes[phase]))
                        epoch_batch_counting = epoch_batch_counting +1
                        # outputs_sigmoid_np = outputs_sigmoid.data.cpu().numpy()
                        # masks_np = masks.data.cpu().numpy()
                        # acc, roc, jac, recall, precision, f1, aji = ComputeMetrics_Batch(outputs_sigmoid_np, masks_np)
                        # running_F1 += f1
                        # running_AJI += aji


                # statistics
                running_loss += total_loss.item() * inputs.size(0)
                running_Nuclei_DC += get_DC(outputs_sigmoid[:,0,:,:],masks[:,0,:,:]) * inputs.size(0)
                running_Border_DC += get_DC(outputs_sigmoid[:,1,:,:],masks[:,1,:,:]) * inputs.size(0)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_Nuclei_DC = running_Nuclei_DC / dataset_sizes[phase]
            epoch_Border_DC = running_Border_DC / dataset_sizes[phase]
            epoch_DC = (epoch_Nuclei_DC+epoch_Border_DC)/2
            print('{} Total_Loss: {:.4f}, Nuclei_DC: {:.4f}, Border_DC: {:.4f}'.format(phase, epoch_loss, epoch_Nuclei_DC, epoch_Border_DC))

            # deep copy the model
            if phase == 'Train':
                train_logs['train_loss'].append(epoch_loss)
                train_logs['train_dc'].append(epoch_DC)

            else:
                # epoch_F1 = running_F1 / dataset_sizes[phase]
                # epoch_AJI = running_AJI / dataset_sizes[phase]
                # print('{} F1: {:.4f}, AJI: {:.4f}'.format(phase, epoch_F1, epoch_AJI))

                train_logs['val_loss'].append(epoch_loss)
                train_logs['val_dc'].append(epoch_DC)
                # train_logs['val_f1'].append(epoch_F1)
                # train_logs['val_aji'].append(epoch_AJI)

                for param_group in optimizer.param_groups:
                    train_logs['lr'].append(param_group['lr'])

                Train_logs_Save_Path = os.path.join(Weight_Save_Dir, 'train_logs.csv')

                df = pd.DataFrame(train_logs)
                df.to_csv(Train_logs_Save_Path, encoding='utf-8')

                if epoch_DC > best_DC or epoch % 10 == 0:
                    best_DC = epoch_DC
                    Weight_Save_Path = os.path.join(Weight_Save_Dir,'epoch_{}_dc_{:.2f}.pth'.format(epoch,epoch_DC))

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss
                    }, Weight_Save_Path)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val DC: {:4f}'.format(best_DC))
    return model


def main():
    Test_Tile_Size = 500
    Train_Tile_Size = 512

    Root_Dir = r'F:\Projects\HE_IHC_Nuclei_Identification\Data'
    Train_Img_Dir = Root_Dir + r'\Tiles\Monuseg\images\train_tiles'
    Train_Mask_Dir = Root_Dir + r'\Tiles\Monuseg\mask_border_v2\train_tiles'
    Test_Img_Dir = Root_Dir + r'\Tiles\Monuseg\images\test_tiles'
    Test_Mask_Dir = Root_Dir + r'\Tiles\Monuseg\mask_border_v2\test_tiles'

    Deep_Model = 'Unet_1' ## Unet_1, Unet_2
    Postfix = 'Border_Loss'
    Weight_Save_Dir = Root_Dir + r'\Weights\Monuseg\{}_{}_{}'.format(Deep_Model, Train_Tile_Size, Postfix)

    Weight_Load_Flag = False
    Weight_Load_Path =  Root_Dir + r'\Weights\Monuseg\Unet_1_512_\epoch_5_dc_0.11.pth'
    Batch_Size = 2
    Num_Workers = 8
    Learning_Rate = 0.0001

    Class_Dict = CLASS_ID_NAME_Nuclei_Seg

    if not os.path.exists(Weight_Save_Dir):
        os.makedirs(Weight_Save_Dir)

    Datasets_Train, DataLoader_Train = get_loader(Train_Img_Dir, Train_Mask_Dir, Img_Size=Train_Tile_Size, Mode='Train', Batch_Size=Batch_Size, Num_Workers=Num_Workers)
    Datasets_Val, DataLoader_Val = get_loader(Test_Img_Dir, Test_Mask_Dir, Img_Size=Train_Tile_Size, Mode='Val', Batch_Size=Batch_Size, Num_Workers=Num_Workers)


    dataset_sizes = {'Train': len(Datasets_Train),
                     'Val': len(Datasets_Val)}

    dataloaders = {'Train': DataLoader_Train,
                    'Val': DataLoader_Val}

    Class_Num = len(Class_Dict) #No Yes

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if Deep_Model == 'Unet_1': #batch size = 64
        model = UNet_1(in_channels=1, n_classes=2, depth=5, wf=6, padding=True,  batch_norm=True, up_mode='upsample')

    elif Deep_Model == 'Unet_2':
        model = UNet_2(img_ch=1, output_ch=2)
    else:
        print('Wrong Deep Learning Model!!!!')
        exit(0)

    # model_ft.load_state_dict(torch.load(Weight_Load_Path))
    optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
    epoch_load=1
    if Weight_Load_Flag:
        checkpoint = torch.load(Weight_Load_Path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_load = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Load Weights: {}'.format(os.path.split(Weight_Load_Path)[-1]))
    else:
        print('Train from scratch')

    # model = nn.DataParallel(model)
    model = model.to(device)

    # if validation loss does not reduce, turn down learning rate by facter 0.1
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size= int(len(Datasets_Train)/Batch_Size*2), gamma=0.5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= int(len(Datasets_Train)/Batch_Size*1000), gamma=0.99)

    train_model(model, dataloaders, dataset_sizes, optimizer, exp_lr_scheduler,
                           Weight_Save_Dir, device, epoch_load=epoch_load, num_epochs=200)

if __name__ == '__main__':
    main()
