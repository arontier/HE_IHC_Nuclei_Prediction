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
import imgaug
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage import morphology
from skimage import img_as_uint
from skimage.morphology import watershed


class RandomRotate(object):

    def __call__(self, img):
        dispatcher = {
            0: img.transpose(PIL.Image.ROTATE_90),
            1: img.transpose(PIL.Image.ROTATE_180),
            2: img.transpose(PIL.Image.ROTATE_270)
        }
        return dispatcher[random.randint(0, 2)]  # randint is inclusive

class ImgGeoTransform(object):
    def __init__(self,translate_px = (-10, 10), rotate = (-10,10), shear = (-10,10), order = 1, cval= 0, fit_output = False, mode = imgaug.ALL):
        self.aug = iaa.Sequential([
            iaa.Affine(translate_px=translate_px, rotate=rotate, shear=shear, order=order, cval=cval, fit_output=fit_output, mode=mode),
        ])

    def __call__(self, img):

        img = np.array(img)
        img = img.astype(np.uint8)
        img = self.aug.augment_image(img)
        img = Image.fromarray(img)

        return img

class ImgGaussBlur(object):
    def __init__(self, probability=0.5, sigma=(1,5)):
        self.aug = iaa.Sequential([
            iaa.Sometimes(probability, iaa.GaussianBlur(sigma=sigma)),
        ])

    def __call__(self, img):

        img = np.array(img)
        img = img.astype(np.uint8)
        img = self.aug.augment_image(img)
        img = Image.fromarray(img)

        return img



class Color_Deconvolution(object):

    def __call__(self, img):
        img = np.array(img)
        img = img.astype(np.uint8)

        if img.ndim != 1:
            I_Deconv = rgb2hed(img)
            Hema = I_Deconv[:, :, 0]
            Hema = rescale_intensity(Hema, out_range=(0, 1))
            Hema = Hema * 255
            Hema = np.uint8(Hema)
        else:
            Hema = img

        Hema = Image.fromarray(Hema)

        return Hema