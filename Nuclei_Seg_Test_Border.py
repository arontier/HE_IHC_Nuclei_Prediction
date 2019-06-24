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
from Evaluation_Metrics_Border import *


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

def test_model(model, dataloaders, dataset_sizes, Pred_Save_Dir, device="cuda:0"):

    model.eval()

    sum_DC = 0
    sum_JAC = 0
    sum_F1 = 0
    sum_AJI = 0

    for i, (inputs, masks, inputs_ori, img_name) in enumerate(dataloaders):

        inputs = inputs.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs_sigmoid = F.sigmoid(outputs)

            print('Validation Steps: {}/{}'.format(i*inputs.size()[0],dataset_sizes))
            # outputs_sigmoid = torch.round(outputs_sigmoid)
            outputs_sigmoid_np = outputs_sigmoid.data.cpu().numpy()
            masks = torch.round(masks)
            masks_np = masks.data.cpu().numpy()
            inputs_ori_np = inputs_ori.data.cpu().numpy()


            acc, roc, jac, recall, precision, f1, aji = ComputeMetrics_Batch(outputs_sigmoid_np, masks_np, inputs_ori=inputs_ori_np, save_dir=Pred_Save_Dir, img_name=img_name)
            sum_DC += get_DC(outputs_sigmoid,masks) * inputs.size(0)
            sum_JAC += jac
            sum_F1 += f1
            sum_AJI += aji

    Mean_DC = sum_DC / dataset_sizes
    Mean_JAC = sum_JAC / dataset_sizes
    Mean_F1 = sum_F1 / dataset_sizes
    Mean_AJI = sum_AJI / dataset_sizes

    print('Mean_DC : {:.4f} Mean_JAC: {:.4f}, Mean_F1: {:.4f}, Mean_AJI: {:.4f}'.format(Mean_DC, Mean_JAC, Mean_F1, Mean_AJI))


def main():
    Test_Tile_Size = 500
    Train_Tile_Size = 512

    Root_Dir = r'F:\Projects\HE_IHC_Nuclei_Identification\Data'

    Test_Img_Dir = Root_Dir + r'\Tiles\Monuseg\images\test_tiles'
    Test_Mask_Dir = Root_Dir + r'\Tiles\Monuseg\mask_v2\test_tiles'

    # Test_Img_Dir = r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\190523_tmp_tiles\Hema'
    # Test_Mask_Dir = r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\190523_tmp_tiles\Hema'


    Deep_Model = 'Unet_1' ## Unet_1, Unet_2

    Weight_Test_Path =  r'F:\Projects\HE_IHC_Nuclei_Identification\Data\Weights\Monuseg\Unet_1_512_Border\epoch_25_dc_0.62.pth'
    Pred_Save_Dir = r'F:\Projects\HE_IHC_Nuclei_Identification\Data\Results\Monuseg\Pred_Image_Border_Best_PostProcess_35_0.5_0.5528'
    Batch_Size = 2
    Num_Workers = 6
    Learning_Rate = 0.0001

    if not os.path.exists(Pred_Save_Dir):
        os.makedirs(Pred_Save_Dir)

    Class_Dict = CLASS_ID_NAME_Nuclei_Seg
    Class_Num = len(Class_Dict)

    Datasets_Val, DataLoader_Val = get_loader(Test_Img_Dir, Test_Mask_Dir, Img_Size=Train_Tile_Size, Mode='Test', Batch_Size=Batch_Size, Num_Workers=Num_Workers)
    dataset_sizes = len(Datasets_Val)
    dataloaders = DataLoader_Val

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if Deep_Model == 'Unet_1': #batch size = 64
        model = UNet_1(in_channels=1, n_classes=2, depth=5, wf=6, padding=True,  batch_norm=True, up_mode='upsample')
    elif Deep_Model == 'Unet_2':
        model = UNet_2(img_ch=1, output_ch=2)
    else:
        print('Wrong Deep Learning Model!!!!')
        exit(0)

    checkpoint = torch.load(Weight_Test_Path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Load Weights: {}'.format(os.path.split(Weight_Test_Path)[-1]))

    # model = nn.DataParallel(model)
    model = model.to(device)

    test_model(model, dataloaders, dataset_sizes, Pred_Save_Dir, device)

if __name__ == '__main__':
    main()
