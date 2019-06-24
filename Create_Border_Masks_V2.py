import sys
from os import path, mkdir, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2
import timeit
from skimage import measure
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed
import os
from matplotlib import pyplot as plt

def create_mask(labels_folder, img_id, masks_out_folder):
    labels_ori = cv2.imread(path.join(labels_folder, '{0}.tif'.format(img_id)), cv2.IMREAD_UNCHANGED)
    labels = measure.label(labels_ori)
    tmp = dilation(labels > 0, square(9))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))
    border = (255 * tmp).astype('uint8')

    channel3_dump = np.zeros_like(labels, dtype='uint8')

    # msk = np.stack((labels_ori, border, channel3_dump))
    msk = np.stack((labels_ori, border, channel3_dump))

    msk = np.rollaxis(msk, 0, 3)
    cv2.imwrite(path.join(masks_out_folder, '{0}.tif'.format(img_id)), msk)

    return 0


if __name__ == '__main__':


    masks_out_folder = r'F:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\Monuseg\mask_border'
    labels_folder = r'F:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\Monuseg\label_border_v2'

    Label_Names = [f[:-4] for f in os.listdir(labels_folder) if f.endswith('.tif')]

    for Label_Name in Label_Names:
        create_mask(labels_folder, Label_Name, masks_out_folder)