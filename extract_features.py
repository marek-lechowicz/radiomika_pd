import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

data_path = r'./data'
data_files = os.listdir(data_path)

# features data frame
features = None

# Instantiate the extractor
settings = {'label': 255, 'force2D': True}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)
i = 0
for file in data_files:
    
    if i > 10:
        break
    filename = file.split('.')[0]
    if "mask" in filename:
        continue
    else: 
        mask_filename = filename + "_mask"

    print(f"Processing: {file}")
    i += 1
    mask_path = data_path  + "/" + mask_filename + ".tif"
    img_path = data_path  + "/" + filename + ".tif"

    result = extractor.execute(img_path, mask_path)
    to_save = dict()
    for key, value in result.items():
        if any(feature_type in key for feature_type in extractor.enabledFeatures.keys()):
            to_save[key] = value
    if features is None:
        features = pd.DataFrame([result])
    else:
        features = features.append(result, ignore_index=True)

features.to_csv("features.csv")
       