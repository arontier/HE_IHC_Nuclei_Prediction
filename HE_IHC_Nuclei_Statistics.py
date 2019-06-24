import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color
import skimage
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 10
# plt.rcParams['image.cmap'] = 'gray'
titlesize = 24

def Plot_Nuclei_Bounding_Box(Img, objProps, Title, ax, Img_Deconv=None):
    score = []
    ax.imshow(Img)
    ax.set_xlim([0, Img.shape[0]])
    ax.set_ylim([Img.shape[1], 0])
    ax.set_title(Title, fontsize=titlesize)
    if Img_Deconv is not None:
        Img_Deconv = Img_Deconv / 255

    for i in range(len(objProps)):
        c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
        width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
        height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1

        cur_bbox = {
            "type": "rectangle",
            "center": c,
            "width": width,
            "height": height,
        }
        tmp_x = c[0] - 0.5 * width
        tmp_y = c[1] - 0.5 * height
        ax.plot(c[0], c[1], 'g+')
        mrect = mpatches.Rectangle([tmp_x, tmp_y],
                                   width, height, fill=False, ec='g', linewidth=1)
        ax.add_patch(mrect)
        if Img_Deconv is not None:

            tmp_x = int(tmp_x)
            tmp_y = int(tmp_y)
            if tmp_y<0:
                tmp_y=0
            if tmp_x < 0:
                tmp_x = 0
            tmp_Img_Deconv = Img_Deconv[tmp_y : tmp_y+height, tmp_x : tmp_x+width]

            tmp_score = 1-np.mean(tmp_Img_Deconv)
            caption = "{:.2f}".format(tmp_score)
            ax.text(tmp_x, tmp_y, caption,
                    color='black', size=8, backgroundcolor="none")
            score.append(tmp_score)
    return score

# plt.imshow(np.array(tmp_Img_Deconv*255).astype(np.uint8))
def Color_Deconvolution(Img, Channel):

    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB

    Img = rgb2hed(Img)
    Img = Img[:, :, Channel]
    Img = rescale_intensity(Img, out_range=(0, 1))
    Img = 1 - Img
    Img = np.uint8(Img*255)

    return Img

if __name__ == '__main__':

    input_image_file = r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\190523_tmp_tiles\Hema\4336_23744_Target.jpg'
    input_image_file_2 = r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\190523_tmp_tiles\Prrx1\4336_23744_Moving.jpg'

    im_input = skimage.io.imread(input_image_file)[:, :, :3]
    im_input_2 = skimage.io.imread(input_image_file_2)[:, :, :3]


    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }
    # segment foreground
    foreground_threshold = 135 #60
    # run adaptive multi-scale LoG filter
    min_radius = 3
    max_radius = 5
    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 3
    # filter out small objects
    min_nucleus_area = 25

    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains

    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    # im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input, W).Stains
    # get nuclei/hematoxylin channel
    # im_nuclei_stain = im_stains[:, :, 0]

    im_nuclei_stain = Color_Deconvolution(im_input, 0)

    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < foreground_threshold)

    # # Display results
    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 3, 1)
    # plt.imshow(im_input)
    # _ = plt.title('Input Image', fontsize=16)
    # plt.subplot(1, 3, 2)
    # plt.imshow(im_stains[:, :, 0])
    # plt.title(stain_1, fontsize=titlesize)
    # plt.subplot(1, 3, 3)
    # plt.imshow(im_fgnd_mask)
    # _ = plt.title('Mask', fontsize=titlesize)

    im_log_max, im_sigma_max = htk.filters.shape.cdog(
        im_nuclei_stain, im_fgnd_mask,
        sigma_min=min_radius * np.sqrt(2),
        sigma_max=max_radius * np.sqrt(2)
    )

    im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
        im_log_max, im_fgnd_mask, local_max_search_radius)

    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

    print('Number of nuclei = ', len(objProps))

    fig, axs = plt.subplots(1,2)
    Plot_Nuclei_Bounding_Box(im_input, objProps, 'Hema', axs[0])

    score = Plot_Nuclei_Bounding_Box(im_input_2, objProps, 'Prrx1', axs[1], Img_Deconv=Color_Deconvolution(im_input_2, 2))

    plt.show()