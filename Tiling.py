import numpy as np
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt
import copy
from skimage import io
import math



if __name__ == '__main__':

    Tile_Num_For_One_Image = 25 ## x**2
    Train_Tile_Size = 512
    Test_Tile_Size = 500
    Train_Tag = 1 ### if tiling train images
    
    Root_Dir = r'F:\Projects\HE_IHC_Nuclei_Identification\Data'
    Img_Dir = Root_Dir + r'\Tiles\Monuseg\images\train'
    Mask_Dir = Root_Dir + r'\Tiles\Monuseg\mask_v2\train'

    Img_Tiling_Dir = Img_Dir + r'_tiles'
    Mask_Tiling_Dir = Mask_Dir + r'_tiles'

    Img_Names = [f for f in os.listdir(Img_Dir)]


    for Img_Name in Img_Names:
        print(Img_Name)
        Img_Path = os.path.join(Img_Dir, Img_Name)
        Mask_Path = os.path.join(Mask_Dir, Img_Name.split(".")[0]+'.tif')

        Img = cv2.imread(Img_Path)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

        Mask = cv2.imread(Mask_Path)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2RGB)

        if Train_Tag:
            Sampling_Row_Length = Img.shape[0] - Test_Tile_Size
            Sampling_Col_Length = Img.shape[1] - Test_Tile_Size
        else:
            Sampling_Row_Length = Img.shape[0]
            Sampling_Col_Length = Img.shape[1]

        for Row_Index in range(0, Sampling_Row_Length, int(Sampling_Row_Length//math.sqrt(Tile_Num_For_One_Image))):
            for Col_Index in range(0, Sampling_Col_Length, int(Sampling_Col_Length//math.sqrt(Tile_Num_For_One_Image))):
                tmp_Img = Img[Row_Index:Row_Index+Test_Tile_Size, Col_Index:Col_Index+Test_Tile_Size, :]
                tmp_Mask = Mask[Row_Index:Row_Index+Test_Tile_Size, Col_Index:Col_Index+Test_Tile_Size, :]

                tmp_Img_Tile_Save_Path = os.path.join(Img_Tiling_Dir, str(Row_Index)+'_'+str(Col_Index) +'_'+Img_Name.split(".")[0]+'.jpg')
                tmp_Mask_Tile_Save_Path = os.path.join(Mask_Tiling_Dir, str(Row_Index)+'_'+str(Col_Index) +'_'+Img_Name.split(".")[0]+'.jpg')

                io.imsave(tmp_Img_Tile_Save_Path, tmp_Img)
                io.imsave(tmp_Mask_Tile_Save_Path, tmp_Mask)

