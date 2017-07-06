# - coding: utf-8 -*-
# python 3.4+

import numpy as np
import cv2
import os
import setting
import logging


def cut(img, size):
    h = img.shape[0]
    w = img.shape[1]
    if h > w:
        return cv2.resize(img[:w, ], (size, size))
    else:
        return cv2.resize(img[:, :h], (size, size))


def get_image_list(dir):
    file_list = os.listdir(dir)
    img_list = []
    for file in file_list:
        if file.endswith(('jpg', 'jpeg', 'bmp', 'png', 'JPG', 'JPEG', 'BMP', 'PNG')):
            img_list.append(file)
    return img_list


def create_data_set(dir, size):
    logging.info("start create data set...")
    img_list = get_image_list(dir=dir)
    n = len(img_list)
    data_set = np.zeros((n, 3, size, size), dtype=np.uint8)
    logging.info("get %d images.." % n)
    for i in range(n):
        logging.debug("read img: %d/%d" % (i, n))
        img = cv2.imread(dir + img_list[i])
        img = cut(img, size=size)
        data_set[i, 0] = img[:, :, 0]
        data_set[i, 1] = img[:, :, 1]
        data_set[i, 2] = img[:, :, 2]
    logging.info("end create data set !")
    return data_set


if __name__ == "__main__":
    data_dir = setting.DATA_DIR
    if not data_dir.endswith('/'):
        data_dir += '/'
    dataset = create_data_set(dir=data_dir, size=setting.IMG_SIZE)
    np.save('dataset.npy', dataset)
    logging.info("data set saved!")
