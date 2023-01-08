import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.model_selection import train_test_split
# set path for image data
read_path = r'./data'
write_path = r'./data_dl'

train_malign = f'{write_path}/train/malign'
train_benign = f'{write_path}/train/benign'

test_malign = f'{write_path}/test/malign'
test_benign = f'{write_path}/test/benign'

val_malign = f'{write_path}/val/malign'
val_benign = f'{write_path}/val/benign'

paths = [write_path, train_malign, train_benign, test_malign, test_benign, val_malign, val_benign]

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

data_files = os.listdir(read_path)

# get clinical data
clinical = pd.read_csv(r'./raw_data/data.csv')

random_state = 44
X = list()
y = list()

for file in data_files:
    # Check if file is a mask file 
    filename = file.split('.')[0]
    # if yes - skip it
    if "mask" in filename:
        continue

    img_path = read_path  + "/" + filename + ".tif"

    # inform which file is currently processed (to track the progress)
    print(f"Processing: {file}")

    # get patient name
    arr_filename = filename.split('_')
    patient_name = arr_filename[0] + "_" + arr_filename[1] + "_" + arr_filename[2]
    
    # get information if cancer was malign
    malign = clinical[clinical['Patient'] == patient_name]['death01'].values[0]

    X.append(filename)
    y.append(malign)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)

OLD_FILE_EXTENTION = '.tif'
NEW_FILE_EXTENTION = '.jpg'

print(f"Saving TRAIN images")
for file, malign in zip(X_train, y_train):
    img = cv2.imread(f'{read_path}/{file}{OLD_FILE_EXTENTION}')
    if malign == 1:
        cv2.imwrite(f'{train_malign}/{file}{NEW_FILE_EXTENTION}', img)
    elif malign == 0:
        cv2.imwrite(f'{train_benign}/{file}{NEW_FILE_EXTENTION}', img)

print(f"Saving VALIDATION images")
for file, malign in zip(X_test, y_test):
    img = cv2.imread(f'{read_path}/{file}{OLD_FILE_EXTENTION}')
    if malign == 1:
        cv2.imwrite(f'{val_malign}/{file}{NEW_FILE_EXTENTION}', img)
    elif malign == 0:
        cv2.imwrite(f'{val_benign}/{file}{NEW_FILE_EXTENTION}', img)

print(f"Saving TEST images")
for file, malign in zip(X_val, y_val):
    img = cv2.imread(f'{read_path}/{file}{OLD_FILE_EXTENTION}')
    if malign == 1:
        cv2.imwrite(f'{test_malign}/{file}{NEW_FILE_EXTENTION}', img)
    elif malign == 0:
        cv2.imwrite(f'{test_benign}/{file}{NEW_FILE_EXTENTION}', img)