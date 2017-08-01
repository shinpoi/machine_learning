# - coding: utf-8 -*-
# python 3.4+

import numpy as np
import cv2
import os
import setting
import logging

batch_size = setting.BATCH
channel = setting.CHANNEL


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
    n = len(img_list)
    n -= n % batch_size
    return img_list[:n]


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
        data_set[i, ] = img.transpose(1, 2, 0)
    logging.info("end create data set !")
    return data_set


def create_data_set_gray(dir, size):
    logging.info("start create data set (gray)...")
    img_list = get_image_list(dir=dir)
    n = len(img_list)
    data_set = np.zeros((n, 1, size, size), dtype=np.uint8)
    logging.info("get %d images.." % n)
    for i in range(n):
        logging.debug("read img: %d/%d" % (i, n))
        img = cv2.imread(dir + img_list[i], 0)
        img = cut(img, size=size)
        data_set[i, 0, ] = img
    logging.info("end create data set !")
    return data_set


if __name__ == "__main__":
    data_dir = setting.DATA_DIR
    if not data_dir.endswith('/'):
        data_dir += '/'
    if channel == 3:
        dataset = create_data_set(dir=data_dir, size=setting.IMG_SIZE)
        save_name = 'dataset_color.npy'
    elif channel == 1:
        dataset = create_data_set_gray(dir=data_dir, size=setting.IMG_SIZE)
        save_name = 'dataset_gray.npy'
    else:
        logging.error("invalid channel: %d" % channel)
        raise ValueError
    np.save(save_name, dataset)
    logging.info("data set saved!")
