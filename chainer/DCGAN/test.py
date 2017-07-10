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
test_num = 20

logging.info('load model...')
generator = model.Generator()
serializers.load_npz('model_gen.npz', generator)

logging.info('generate images...')
x = np.random.uniform(0, 1, (test_num, input_num))
with no_backprop_mode():
    x = Variable(np.array(x, dtype=np.float32))
    with using_config('train', False):
        imgs = (np.array(generator(x).data)+2) * 20

logging.info('save images...')
for i in range(test_num):
    logging.debug("save image: %d/%d" % (i, test_num))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img_ = imgs[i, ]
    for c in range(3):
        img[:, :, c] = np.array(img_[c, ], dtype=np.uint8)

    cv2.imwrite("gen_" + str(i) + '.jpg', img)

logging.info('finished !')

