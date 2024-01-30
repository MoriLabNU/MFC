# ---------------------------------------------------------------
# prepare the simulated dataset, copy to the dir & add black ground
# ---------------------------------------------------------------
import os
from os.path import join as osj
import shutil
import cv2
import numpy as np
import tqdm

if __name__ == '__main__':

    save_root = './DATA/simulated'
    data_root = './data/Laparoscopic-img2img'

    # 1. copy
    pbar = tqdm.tqdm(
        total=int(20000),
        desc="copying frames",
    )

    names = ['inputs/simulated', 'labels/simulated']
    os.makedirs(osj(save_root, 'images'))
    os.makedirs(osj(save_root, 'labels'))

    for i in os.listdir(osj(data_root, names[0])):
        for j in os.listdir(osj(data_root, names[0], i)):
            for k in os.listdir(osj(data_root, names[0], i, j)):
                name = i + '_' + j + '_' + k
                # print(name)
                shutil.copy(osj(data_root, names[0], i, j, k), osj(save_root, 'images', name))
                shutil.copy(osj(data_root, names[1], i, 'labels', k.replace('img', 'lbl')),
                            osj(save_root, 'labels', name))
                pbar.update(1)
    pbar.close()

    # 2. add_background
    pbar = tqdm.tqdm(
        total=int(20000),
        desc="add background frames",
    )

    folds = ['images', 'labels']

    xx, yy = np.mgrid[:256, :452]
    circle = (xx - 128) ** 2 + (yy - 226) ** 2
    donut = np.logical_not(circle > (178 ** 2))[:, :, np.newaxis]
    bg = np.repeat(donut, 3, axis = 2)

    for i in os.listdir(osj(save_root, 'images')):
        name_bg = i[:-4] + '_bg.png'
        # print(name_bg)
        img = cv2.imread(osj(save_root, 'images', i))
        lb = cv2.imread(osj(save_root, 'labels', i))
        # print(img.shape, lb.shape)
        assert img.shape == lb.shape and lb.shape == (256, 452, 3)
        img *= bg
        lb *= bg

        cv2.imwrite(osj(save_root, 'images', name_bg), img)
        cv2.imwrite(osj(save_root, 'labels', name_bg), lb)
        pbar.update(1)
    pbar.close()
