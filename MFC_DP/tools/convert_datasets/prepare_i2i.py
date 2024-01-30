# ---------------------------------------------------------------
# prepare the i2i dataset, copy to the dir & add black ground
# ---------------------------------------------------------------
import os
from os.path import join as osj
import shutil
import cv2
import numpy as np
import tqdm

if __name__ == '__main__':

    save_root = './DATA/i2i'
    data_root = './data/Laparoscopic-img2img'

    # 1. copy
    pbar = tqdm.tqdm(
        total=int(100000),
        desc="copying frames",
    )

    names = ['styleFromCholec80', 'labels/simulated']
    os.makedirs(osj(save_root, 'images'), exist_ok=True)
    os.makedirs(osj(save_root, 'labels'), exist_ok=True)

    # 2. add_background

    folds = ['images', 'labels']

    xx, yy = np.mgrid[:256, :452]
    circle = (xx - 128) ** 2 + (yy - 226) ** 2
    donut = np.logical_not(circle > (178 ** 2))[:, :, np.newaxis]
    bg = np.repeat(donut, 3, axis=2)

    for i in os.listdir(osj(data_root, names[0])):
        for j in os.listdir(osj(data_root, names[0], i)):
            for k in os.listdir(osj(data_root, names[0], i, j)):
                name = i + '_' + j + '_' + k
                img = cv2.imread(osj(data_root, names[0], i, j, k))
                lb = cv2.imread(osj(data_root, names[1], i, 'labels', k.replace('img', 'lbl')))
                assert img.shape == lb.shape and lb.shape == (256, 452, 3)
                img *= bg
                lb *= bg

                cv2.imwrite(osj(save_root, 'images', name), img)
                cv2.imwrite(osj(save_root, 'labels', name), lb)

                pbar.update(1)
    pbar.close()
