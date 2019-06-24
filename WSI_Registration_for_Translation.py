import openslide
import numpy as np
import cv2
from scipy import misc
import skimage
import skimage.io
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
import math
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_local
from scipy import signal
import random
import matplotlib.patches as patches
import copy
import os

def Color_Deconvolution(Img, Channel):

    # Channel 0 -> Hema
    # Channel 1 -> Eosin
    # Channel 2 -> DAB

    Img = rgb2hed(Img)
    Img = Img[:, :, Channel]
    Img = rescale_intensity(Img, out_range=(0, 1))
    Img = 1-Img
    Img = np.uint8(Img*255)

    return Img

def Registration_Translation(Target_Img, Moving_Img):

    # Target_Img = skimage.color.rgb2gray(Target_Img)*255
    # Moving_Img = skimage.color.rgb2gray(Moving_Img)*255

    Target_Img = Color_Deconvolution(Target_Img, 0)
    Moving_Img = Color_Deconvolution(Moving_Img, 2)


    Target_Img_B = Target_Img
    Moving_Img_B = Moving_Img

    # Target_Img_B = Target_Img > 220
    # Moving_Img_B = Moving_Img > 220

    # adaptive_thresh = threshold_local(Target_Img_B, 55, offset=10)
    # Target_Img_B = Target_Img > adaptive_thresh
    # adaptive_thresh = threshold_local(Moving_Img_B, 55, offset=10)
    # Moving_Img_B = Moving_Img_B > adaptive_thresh

    # plt.figure()
    # plt.imshow(Target_Img_B, cmap='gray')
    # plt.figure()
    # plt.imshow(Moving_Img_B, cmap='gray')
    # plt.show()

    # Calculate cross correlation

    Cross_Corr = signal.fftconvolve(Target_Img_B, Moving_Img_B[::-1, ::-1])
    # print(np.max(Cross_Corr))
    ind = np.unravel_index(np.argmax(Cross_Corr, axis=None), Cross_Corr.shape)

    # Moving_Img_Translation-> + , plus the tranlation to moving image coord , Moving_Img_Translation-> - minus the tranlation to moving image coord
    Moving_Img_Translation = np.array(ind) - np.array(Moving_Img_B.shape)

    return Target_Img_B, Moving_Img_B, Moving_Img_Translation

def Local_Registration_Translation_Candidates(Target_P, Moving_P, Target_Downsampled_B, Downsample_Times, G_Moving_Img_Translation):

    Coordinates_Select_Num = 1
    Target_Img_Size = 750 #2000
    Moving_Img_Size = 500  #500

    listOfCoordinates = np.where(Target_Downsampled_B < 150)

    # plt.imshow(Target_Downsampled_B > 100)
    # plt.show()

    listOfCoordinates = list(zip(listOfCoordinates[0], listOfCoordinates[1]))
    listOfCoordinates = np.array(listOfCoordinates) * Downsample_Times
    listOfCoordinates_Selected = random.choices(listOfCoordinates, k=Coordinates_Select_Num)
    # print(listOfCoordinates_Selected)

    L_Moving_Img_Translation_Candidates = []

    ## row col exchange !!!!!
    for Coordinates in listOfCoordinates_Selected:

        Target_Coordinates = (Coordinates[1]- (Target_Img_Size//2 - Moving_Img_Size//2), Coordinates[0]- (Target_Img_Size//2 - Moving_Img_Size//2))
        Moving_Coordinates = (Coordinates[1] - G_Moving_Img_Translation[1] , Coordinates[0]- G_Moving_Img_Translation[0])

        tmp_Target_Img = np.array(Target_P.read_region(Target_Coordinates, 0, (Target_Img_Size, Target_Img_Size)))[:,:,:3]
        tmp_Moving_Img = np.array(Moving_P.read_region(Moving_Coordinates, 0, (Moving_Img_Size, Moving_Img_Size)))[:,:,:3]


        #############################################################
        ''''' 
       Global Registration
       '''''
        # fig, ax = plt.subplots(1)
        # ax.imshow(tmp_Target_Img)
        # rect = patches.Rectangle(((Target_Img_Size//2 - Moving_Img_Size//2), (Target_Img_Size//2 - Moving_Img_Size//2)), Moving_Img_Size, Moving_Img_Size, linewidth=2, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.figure()
        # plt.imshow(tmp_Moving_Img)
        # tmp_tmp_Target_Img = copy.deepcopy(tmp_Target_Img)
        # tmp_tmp_Target_Img[(Target_Img_Size//2 - Moving_Img_Size//2):(Target_Img_Size//2 - Moving_Img_Size//2)+Moving_Img_Size,\
        #                    (Target_Img_Size//2 - Moving_Img_Size//2):(Target_Img_Size//2 - Moving_Img_Size//2)+Moving_Img_Size, :] = tmp_Moving_Img
        #
        # plt.figure()
        # plt.imshow(tmp_tmp_Target_Img)
        # plt.show()
        ############################################################


        tmp_Target_Img_B, tmp_Moving_Img_B, tmp_Moving_Img_Translation = Registration_Translation(tmp_Target_Img, tmp_Moving_Img)

        tmp_Moving_Img_Translation= tmp_Moving_Img_Translation - (Target_Img_Size - Moving_Img_Size)//2
        print(tmp_Moving_Img_Translation)

        #############################################################
        ''''' 
       Local Registration
       '''''

        # tmp_tmp_Target_Img = copy.deepcopy(tmp_Target_Img)
        # tmp_row_index = (Target_Img_Size - Moving_Img_Size)//2 + tmp_Moving_Img_Translation[0]+1
        # tmp_col_index = (Target_Img_Size - Moving_Img_Size)//2 + tmp_Moving_Img_Translation[1]+1
        #
        # tmp_tmp_Target_Img[tmp_row_index : tmp_row_index + Moving_Img_Size, \
        #                    tmp_col_index : tmp_col_index + Moving_Img_Size, :] = tmp_Moving_Img
        # plt.figure()
        # plt.imshow(tmp_tmp_Target_Img)
        # plt.show()
        #############################################################


        L_Moving_Img_Translation_Candidates.append(tmp_Moving_Img_Translation)

    print(L_Moving_Img_Translation_Candidates)
    return L_Moving_Img_Translation_Candidates
def Reject_Outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]

def Select_L_Moving_Img_Translation(L_Moving_Img_Translation_Candidates):
    L_Moving_Img_Translation_Candidates = np.array(L_Moving_Img_Translation_Candidates)
    rows = L_Moving_Img_Translation_Candidates[:,0]
    cols = L_Moving_Img_Translation_Candidates[:,1]

    rows = Reject_Outliers(rows)
    cols = Reject_Outliers(cols)

    return np.mean(rows), np.mean(cols)

def Show_Translation_Images(Target_P, Moving_P, Target_Downsampled_B, Downsample_Times, G_Moving_Img_Translation, L_Moving_Img_Translation):

    Coordinates_Select_Num = 100
    Target_Img_Size = 750 #2000
    Moving_Img_Size = 500  #500

    listOfCoordinates = np.where(Target_Downsampled_B <150)

    listOfCoordinates = list(zip(listOfCoordinates[0], listOfCoordinates[1]))
    listOfCoordinates = np.array(listOfCoordinates) * Downsample_Times
    listOfCoordinates_Selected = random.choices(listOfCoordinates, k=Coordinates_Select_Num)

    for Coordinates in listOfCoordinates_Selected:

        Target_Coordinates = (Coordinates[1]- (Target_Img_Size//2 - Moving_Img_Size//2), Coordinates[0]- (Target_Img_Size//2 - Moving_Img_Size//2))
        Moving_Coordinates = (Coordinates[1] - G_Moving_Img_Translation[1] , Coordinates[0]- G_Moving_Img_Translation[0])

        tmp_Target_Img = np.array(Target_P.read_region(Target_Coordinates, 0, (Target_Img_Size, Target_Img_Size)))[:,:,:3]
        tmp_Moving_Img = np.array(Moving_P.read_region(Moving_Coordinates, 0, (Moving_Img_Size, Moving_Img_Size)))[:,:,:3]

        ###############################################################################
        ''''' 
       Save Image
       '''''
        tmp_row_index = (Target_Img_Size - Moving_Img_Size)//2 + L_Moving_Img_Translation[0]
        tmp_col_index = (Target_Img_Size - Moving_Img_Size)//2 + L_Moving_Img_Translation[1]

        tmp_tmp_Target_Img = tmp_Target_Img[tmp_row_index : tmp_row_index + Moving_Img_Size, \
                                            tmp_col_index : tmp_col_index + Moving_Img_Size, :]


        tmp_Target_Img_Save_Path = os.path.join(r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\190523_tmp_tiles\Hema', '{}_{}_Target.jpg'.format(Coordinates[1],Coordinates[0]))
        tmp_Moving_Img_Save_Path = os.path.join(r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Tiles\190523_tmp_tiles\Prrx1', '{}_{}_Moving.jpg'.format(Coordinates[1],Coordinates[0]))
        skimage.io.imsave(tmp_Target_Img_Save_Path,tmp_tmp_Target_Img)
        skimage.io.imsave(tmp_Moving_Img_Save_Path,tmp_Moving_Img)

        ###############################################################################


        #############################################################
        ''''' 
       Global Registration
       '''''
        # fig, ax = plt.subplots(1)
        # ax.imshow(tmp_Target_Img)
        # rect = patches.Rectangle(((Target_Img_Size//2 - Moving_Img_Size//2), (Target_Img_Size//2 - Moving_Img_Size//2)), Moving_Img_Size, Moving_Img_Size, linewidth=2, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.figure()
        # plt.imshow(tmp_Moving_Img)
        # tmp_tmp_Target_Img = copy.deepcopy(tmp_Target_Img)
        # tmp_tmp_Target_Img[(Target_Img_Size//2 - Moving_Img_Size//2):(Target_Img_Size//2 - Moving_Img_Size//2)+Moving_Img_Size,\
        #                    (Target_Img_Size//2 - Moving_Img_Size//2):(Target_Img_Size//2 - Moving_Img_Size//2)+Moving_Img_Size, :] = tmp_Moving_Img
        #
        # plt.figure()
        # plt.imshow(tmp_tmp_Target_Img)
        # # plt.show()
        ############################################################


        #############################################################
        ''''' 
       Local Registration
       '''''
        # tmp_tmp_Target_Img = copy.deepcopy(tmp_Target_Img)
        # tmp_row_index = (Target_Img_Size - Moving_Img_Size)//2 + L_Moving_Img_Translation[0]
        # tmp_col_index = (Target_Img_Size - Moving_Img_Size)//2 + L_Moving_Img_Translation[1]
        #
        # tmp_tmp_Target_Img[tmp_row_index : tmp_row_index + Moving_Img_Size, \
        #                    tmp_col_index : tmp_col_index + Moving_Img_Size, :] = tmp_Moving_Img
        # plt.figure()
        # plt.imshow(tmp_tmp_Target_Img)
        # plt.show()
        #############################################################



if __name__ == '__main__':

    Downsample_Times = 32

    Target_P = openslide.open_slide(r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Slides\multiplex_IHC\HE_One.svs')
    Moving_P = openslide.open_slide(r'D:\Projects\HE_IHC_Nuclei_Identification\Data\Slides\multiplex_IHC\Prrx1_Three.svs') # Prrx1_Three , SMA_Eight, TNC_Four, Twist1_Two

    Level = int(math.log2(Downsample_Times))-1

    if Level > len(Target_P.level_dimensions):
        Level = len(Target_P.level_dimensions)-1

    Downsample_Times = math.ceil(Target_P.level_dimensions[0][0] // Target_P.level_dimensions[Level][0])
    print(f"Downsample Times: {Downsample_Times}")

    Target_Downsampled = np.array(Target_P.read_region((0, 0), Level, Target_P.level_dimensions[Level]))[:,:,:3]
    Moving_Downsampled = np.array(Moving_P.read_region((0, 0), Level, Moving_P.level_dimensions[Level]))[:,:,:3]

    Target_Downsampled_B, Moving_Downsampled_B, G_Moving_Img_Translation = Registration_Translation(Target_Downsampled, Moving_Downsampled)
    G_Moving_Img_Translation = G_Moving_Img_Translation * Downsample_Times
    print(f"Global Moving Image Translation: {G_Moving_Img_Translation}")

    L_Moving_Img_Translation_Candidates= Local_Registration_Translation_Candidates(Target_P, Moving_P, Target_Downsampled_B, Downsample_Times, G_Moving_Img_Translation)

    L_Moving_Img_Translation = Select_L_Moving_Img_Translation(L_Moving_Img_Translation_Candidates)

    print(f"Local Moving Image Translation: {L_Moving_Img_Translation}")

    L_Moving_Img_Translation = np.array([13, 8])
    Final_Moving_Img_Translation = G_Moving_Img_Translation + L_Moving_Img_Translation

    print(f"Final Moving Image Translation: {Final_Moving_Img_Translation}")

    Show_Translation_Images(Target_P, Moving_P, Target_Downsampled_B, Downsample_Times, G_Moving_Img_Translation, L_Moving_Img_Translation)

