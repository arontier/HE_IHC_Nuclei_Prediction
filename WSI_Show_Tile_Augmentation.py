import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset
import PIL
from Augmentation import RandomRotate, Color_Deconvolution, ImgGeoTransform, ImgGaussBlur
import random
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from Nuclei_Seg_Data_Loader import get_loader

import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

# Helper function to show a batch
def show_batch(inputs):
    """Show image with landmarks for a batch of samples."""


    grid = utils.make_grid(inputs,padding=10)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ =='__main__':
    Train_Tile_Size = 512
    Batch_Size = 1
    Num_Workers = 2  ## Using Arontier-HYY computer, 8 cores are faster than 16 cores
    Root_Dir = r'F:\Projects\HE_IHC_Nuclei_Identification\Data'
    Train_Img_Dir = Root_Dir + r'\Tiles\Monuseg\images\train_tiles'
    Train_Mask_Dir = Root_Dir + r'\Tiles\Monuseg\mask_1\train_tiles'



    dataloader = get_loader(Train_Img_Dir, Train_Mask_Dir, Img_Size=512, Mode='Train', Batch_Size=1, Num_Workers=1)

    for i, (inputs, masks) in enumerate(dataloader):
        plt.figure()
        show_batch(inputs)
        plt.figure()
        show_batch(masks)
        # plt.axis('off')
        plt.ioff()
        plt.show()