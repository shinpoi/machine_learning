# - coding: utf-8 -*-
# python 3.4+

from chainer import Variable, serializers, using_config, no_backprop_mode
import logging
import setting
import model
import numpy as np
import cv2

input_num = setting.RAND_IN_NUM
size = setting.IMG_SIZE
interval = 10
# test_num = 20
channel = 3

logging.info('load model...')
generator = model.GeneratorFace()
# serializers.load_npz('model_gen_6x100.npz', generator)


# flatten histogram
def hist_flatten(src):
    shape = src.shape
    img_new = np.array(src).reshape((-1,))
    hist = cv2.calcHist((img_new,), (0,), None, (256,), (0, 256))
    n = float(len(img_new))

    # normalize
    for i in range(len(hist)):
        hist[i] /= n

    # accumulate
    hist_sum = np.zeros(len(hist))
    for i in range(len(hist)):
        hist_sum[i] = sum(hist[:i])

    # flatten
    for i in range(len(img_new)):
        img_new[i] = 255 * hist_sum[img_new[i]]
    return img_new.resize(shape)


def clip_img(x):
    return np.float32(0 if x < 0 else (1 if x > 1 else x))

"""
logging.info('generate images...')
x = np.random.uniform(-1, 1, (test_num, input_num))
with no_backprop_mode():
    x = Variable(np.array(x, dtype=np.float32))
    with using_config('train', False):
        imgs = (np.array(generator(x).data)+2) * 20

logging.info('save images...')
for i in range(test_num):
    logging.debug("save image: %d/%d" % (i, test_num))
    img = np.zeros((size, size, channel), dtype=np.uint8)
    img_ = imgs[i, ]
    for c in range(channel):
        img[:, :, c] = np.array(img_[c, ], dtype=np.uint8)

    cv2.imwrite("gen_" + str(i) + '.jpg', img)
"""
x = np.random.uniform(0, 1, (64, input_num))
x = Variable(np.array(x, dtype=np.float32))
for i in range(27):
    serializers.load_npz('./model_gen_%dx50.npz' % i, generator)
    logging.info('generate images (big)...')
    img = np.zeros((size*8 + interval*9, size*8 + interval*9, channel), dtype=np.uint8)
    with using_config('train', False):
        with no_backprop_mode():
            imgs = np.array(generator(x).data)

    counter = 0
    for row in range(8):
        for col in range(8):
            t = (np.vectorize(clip_img)(imgs[counter, ]).transpose(1, 2, 0))*255
            # t = cv2.cvtColor(np.array(t, dtype=np.uint8), cv2.COLOR_RGB2HSV)
            # t[:, :, 2] = cv2.equalizeHist(t[:, :, 2])
            # t = cv2.cvtColor(t, cv2.COLOR_HSV2RGB)
            img[row*size+(row+1)*interval:(row+1)*size+(row+1)*interval,
                col*size+(col+1)*interval:(col+1)*size+(col+1)*interval, ] = t
            counter += 1

    cv2.imwrite("gen_%dx64.jpg" % i, img)
    logging.info('save images (big)...')

    logging.info('finished !')

