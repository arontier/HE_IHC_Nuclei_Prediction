import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms, utils

from PIL import Image
import PIL
import random
from imgaug import augmenters as iaa
import imgaug
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

from Color_Deconvolution import Color_Deconvolution

# Helper function to show a batch
def show_batch(inputs):
    """Show image with landmarks for a batch of samples."""


    grid = utils.make_grid(inputs,padding=10)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


class Nuclei_Seg_ImageFolder(torch.utils.data.Dataset):
    def __init__(self, Img_Dir, Mask_Dir, iaa_Seq_Geo, iaa_Seq_Color, Mode):
        """Initializes image paths and preprocessing module."""
        self.Img_Dir = Img_Dir
        self.Mask_Dir = Mask_Dir
        self.iaa_Seq_Geo = iaa_Seq_Geo
        self.iaa_Seq_Color = iaa_Seq_Color
        self.Img_Names = [f for f in os.listdir(self.Img_Dir) if f.endswith('.jpg')]
        self.Mode = Mode
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        Img_Name = self.Img_Names[index]
        Img_Path = os.path.join(self.Img_Dir, Img_Name)
        Mask_path = os.path.join(self.Mask_Dir, Img_Name)

        with open(Img_Path, 'rb') as f:
            Img = Image.open(f)
            Img = Img.convert('RGB')
            Img = np.array(Img)
            Img_Ori = Img.copy()
            Color_Deconvolution_Handle = Color_Deconvolution()
            Img = Color_Deconvolution_Handle.colorDeconv(Img)
            Img = Img[:,:,0]
            Img = Img.astype(np.uint8)
            Img = Img[..., np.newaxis]
        with open(Mask_path, 'rb') as f:
            Mask = Image.open(f)
            Mask = np.array(Mask)
            Mask = Mask.astype(np.uint8)

            #### be careful, the third channel maybe changed
            Mask[:,:,0] = Mask[:,:,2]


        # same geometric transform
        seq_imgs_deterministic = self.iaa_Seq_Geo.to_deterministic()
        Img_Ori = seq_imgs_deterministic.augment_image(Img_Ori)
        Img = seq_imgs_deterministic.augment_image(Img)
        Mask = seq_imgs_deterministic.augment_image(Mask)

        #color for image
        Img = self.iaa_Seq_Color(image = Img)
        Img_Ori = self.iaa_Seq_Color(image = Img_Ori)
        # plt.imshow(Img.squeeze(), cmap='gray')
        # plt.show()
        # transform to PIL for pytorch
        Img = Image.fromarray(Img.squeeze())

        if self.Mode in ['Train', 'Val']:
            Mask = Mask[:, :, :2]
        else:
            Mask = Mask[:,:,0].squeeze()

        ToTensor = transforms.ToTensor()
        ToNormalize = transforms.Normalize(mean=[0.5], std=[0.5])

        Img = ToTensor(Img)
        Img = ToNormalize(Img)

        Mask = ToTensor(Mask)
        Mask = torch.round(Mask)
        # show_batch(Mask)

        return Img, Mask, Img_Ori, Img_Name

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.Img_Names)




def get_loader(Img_Dir, Mask_Dir, Img_Size=512, Mode='Train', Batch_Size=4, Num_Workers=4):
    """Builds and returns Dataloader."""

    iaa_Data_Transform = {

        'Train_Geo': iaa.Sequential([
            iaa.Resize(Img_Size),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(k=(0,3)),
            iaa.Crop(percent = (0, 0.1)),
            iaa.PiecewiseAffine(scale=(0,0.05), nb_rows=(5,10), nb_cols=(5,10), order=0, cval=0, mode=imgaug.ALL),
            iaa.Affine(translate_percent =(-0.1, 0.1), rotate=(-10, 10), shear=(-5, 5), order=0, cval=0, fit_output=False, mode=imgaug.ALL),
        ]),

        'Train_Color': iaa.Sequential([
            iaa.Resize(Img_Size),
            iaa.ContrastNormalization((0.5, 1.5)),
            iaa.Multiply((0.75, 1.25)),
            iaa.GaussianBlur(sigma=(0,5))
        ]),
        'Val_Geo': iaa.Sequential([
            iaa.Resize(Img_Size),
        ]),
        'Val_Color': iaa.Sequential([
            iaa.Resize(Img_Size),
        ])


        # 'Val_Geo': iaa.Sequential([
        #     iaa.Crop(125),
        #     iaa.Resize(Img_Size),
        # ]),
        # 'Val_Color': iaa.Sequential([
        #     iaa.Resize(Img_Size),
        # ])

    }

    if Mode == 'Train':
        iaa_Seq_Geo = iaa_Data_Transform['Train_Geo']
        iaa_Seq_Color = iaa_Data_Transform['Train_Color']
        shuffle = True
    else:
        iaa_Seq_Geo = iaa_Data_Transform['Val_Geo']
        iaa_Seq_Color = iaa_Data_Transform['Val_Color']
        shuffle = False

    dataset = Nuclei_Seg_ImageFolder(Img_Dir=Img_Dir, Mask_Dir=Mask_Dir, iaa_Seq_Geo=iaa_Seq_Geo, iaa_Seq_Color=iaa_Seq_Color, Mode=Mode)


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=Batch_Size,
                                  shuffle=shuffle,
                                  num_workers=Num_Workers)
    return dataset, data_loader
