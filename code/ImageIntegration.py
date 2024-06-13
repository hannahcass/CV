import glob
import json
import os

import cv2
import numpy as np

rootpath = os.path.join("..", "..", "data_WiSAR", "data", "test")
folders = [f.path for f in os.scandir(rootpath) if f.is_dir()]

mask_path = os.path.join("..", "..", "data_WiSAR", "data", "mask.png")
mask = cv2.imread(mask_path)
mask = mask.astype(bool)

imgs = np.empty(shape=(7, 10, 1024, 1024, 3))

for datapath in folders[1:]:

    hom_path = os.path.join(datapath, "homographies.json")
    output = os.path.join(rootpath,"integrated", os.path.basename(datapath))
    os.makedirs(output, exist_ok=True)

    img_list = glob.glob(os.path.join(datapath, '**/*.png'), recursive=True)
    #img_list.sort(key=str)

    with open(hom_path) as fp:
        homographies = json.load(fp)
        fp.close()

    for i, image in enumerate(img_list):
        current = cv2.imread(image) * mask
        hom_key = os.path.basename(img_list[i]).split('.')[0]
        homography = np.array(homographies[hom_key])
        imgs[i//10, i%10, ...] = cv2.warpPerspective(current,homography,current.shape[:2])

    for i in range(7):
        integrated = imgs[i, ...].mean(axis=0)
        cv2.imwrite(os.path.join(output, f"{i}.png"), integrated)