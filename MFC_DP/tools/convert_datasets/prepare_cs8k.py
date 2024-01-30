import os
from os.path import join as osj
import shutil
import cv2
import numpy as np


def main():
    save_root = "./DATA/cholec/"
    data_root = './data/CholecSeg8k'

    lists = os.listdir(data_root)
    lists = sorted(lists)
    print(lists, len(lists))
    for i in range(17):
        video = lists[i]
        print(video)
        for j in os.listdir(osj(data_root, video)):
            for k in os.listdir(osj(data_root, video, j)):
                if '_endo.png' in k:
                    name = video + '_' + k
                    k2 = k.replace('.png', '_watershed_mask.png')
                    name2 = video + '_' + k2
                    if i < 5:
                        flod = 'val'
                    else:
                        flod = 'test'
                    os.makedirs(osj(save_root, 'img', flod), exist_ok=True)
                    os.makedirs(osj(save_root, 'gt', flod), exist_ok=True)
                    shutil.copy(osj(data_root, video, j, k), osj(save_root, 'img', flod, name))
                    shutil.copy(osj(data_root, video, j, k2), osj(save_root, 'gt', flod, name2))


if __name__ == '__main__':
    main()
