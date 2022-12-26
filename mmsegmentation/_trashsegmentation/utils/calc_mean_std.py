# 참고 : https://tw0226.tistory.com/20

import cv2
import numpy as np
import glob
from tqdm import tqdm

path = "/opt/ml/input/data/mmseg/images/train"
exts = [".jpg", ".png"]

data_list = []
for ext in exts:
    data_list += glob.glob(path + "/*" + ext)

img_norm = list()
img_std = list()
for data in tqdm(data_list, total=len(data_list)):
    img = cv2.imread(data, cv2.IMREAD_COLOR).astype(np.float32)
    mean, std = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
    img_norm.append(mean)
    img_std.append(std)

print(np.mean(img_norm, axis=0), np.mean(img_std, axis=0))
