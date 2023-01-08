import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import numpy as np
read_path = r'./raw_data/lgg-mri-segmentation/kaggle_3m'
write_path = r'./data'
arr = os.listdir(read_path)

for folder in arr:
    folder_path = read_path + "/" + folder
    if not os.path.isdir(folder_path):
        continue
    arr_files = os.listdir(folder_path)
    for file in arr_files:
        filname = file.split('.')[0]
        if "mask" not in filname:
            mask_filename = filname + "_mask"
        else: 
            mask_filename = filname
        mask = cv2.imread(folder_path  + "/" + mask_filename + ".tif")
        if np.all(np.unique(mask) == 0):
            continue
        print(filname)
        read_file_path = folder_path + "/" + file
        img = cv2.imread(read_file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # save as tiff with the same name but different extension
        cv2.imwrite(f'{write_path}/{file}', gray)