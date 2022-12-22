import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# set path for image data
data_path = r'./data'
data_files = os.listdir(data_path)

# get clinical data
clinical = pd.read_csv(r'./raw_data/data.csv')

# define "features" DataFrame
features = None

# Instantiate the extractor
settings = {'label': 255, 'force2D': True}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableFeatureClassByName('shape2D')
extractor.enableAllImageTypes()

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)

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

    # inform which file is currently processed (to track the progress)
    print(f"Processing: {file}")

    # execute the extraction
    result = extractor.execute(img_path, mask_path)

    # get patient name
    arr_filename = filename.split('_')
    patient_name = arr_filename[0] + "_" + arr_filename[1] + "_" + arr_filename[2]
    
    # get information if cancer was malign
    malign = clinical[clinical['Patient'] == patient_name]['death01'].values[0]
    
    # define information to be saved as one row
    to_save = {'patient_name': patient_name, 'malign': malign}

    # get only features from "result" dictionary
    for key, value in result.items():
        if any(feature_type in key for feature_type in extractor.enabledFeatures.keys()):
            to_save[key] = value

    # append extracted features to "fetures" DataFrame
    if features is None:
        features = pd.DataFrame([to_save])
    else:
        features = features.append(to_save, ignore_index=True)

# save DataFrame to .csv file
features.to_csv("features.csv")
       