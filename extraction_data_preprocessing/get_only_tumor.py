import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os


# set path for image data
data_path = r'./data'
data_files = os.listdir(data_path)

save_path = r'./data_only_tumor'

clinical = pd.read_csv(r'./raw_data/data.csv')

i = 0
for file in data_files:
    # Check if file is a mask file 
    filename = file.split('.')[0]
    # if yes - skip it
    if "mask" in filename:
        continue
    # if not - get mask file
    else: 
        mask_filename = filename + "_mask"

    mask_path = data_path  + "/" + mask_filename + ".tif"
    img_path = data_path  + "/" + filename + ".tif"

    # get patient name
    arr_filename = filename.split('_')
    patient_name = arr_filename[0] + "_" + arr_filename[1] + "_" + arr_filename[2]
    
    # get information if cancer was malign
    malign = clinical[clinical['Patient'] == patient_name]['death01'].values[0]

    # inform which file is currently processed (to track the progress)
    i += 1
    print(f"{i}) Processing: {file}")

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img = np.multiply(img, mask)

    # explicit stating malign == 1 or 0 automatically gets rid of any unwanted values
    if malign == 1:
        # malign_path = os.path.join(save_path + "/" + "malign")
        cv2.imwrite(save_path + "/" + "malign" + "/" + file, img)
    elif malign == 0:
        # benign_path = os.path.join(save_path, "benign")
        cv2.imwrite(save_path + "/" + "benign" + "/" + file, img)
    

