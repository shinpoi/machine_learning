# - coding: utf-8 -*-
# python 3.4+

import numpy as np
from chainer import Variable, optimizers, optimizer, serializers, cuda
import chainer.functions as F
import logging
import setting
import model
import time


##################
log_name = setting.LOG_DIR + 'com_train_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
logging.basicConfig(filename=log_name)

#  import true image set (192x192)
logging.info("start import dataset...")

batch_size = setting.BATCH

data_set = np.array(np.load('./dataset/dataset_gray.npy'), dtype=np.float32)
data_set = data_set[:len(data_set) - (len(data_set) % batch_size)]
data_set /= 255.0
n = len(data_set)
logging.info("get %d true data for training..." % n)

##################
# set model

# load model
logging.info("start load model...")

com = model.CompletionNet()
dis = model.LGDiscriminator()

opt_com = optimizers.Adam(alpha=setting.ADAM_RATE)
opt_com.setup(com)
opt_dis = optimizers.Adam(alpha=setting.ADAM_RATE)
opt_dis.setup(dis)
if setting.WeightDecay:
    opt_com.add_hook(optimizer.WeightDecay(setting.WeightDecay))
    opt_dis.add_hook(optimizer.WeightDecay(setting.WeightDecay))

# Use GPU
if setting.GPU:
    gpu_device = 0
    cuda.get_device_from_id(gpu_device).use()
    com.to_gpu(gpu_device)
    dis.to_gpu(gpu_device)
    xp = cuda.cupy
    # data_set = xp.array(data_set)
    logging.info('GPU used')

##################
#  train
# True = 1, Fake = 0
# Mask: target = 1, others = 0
batch_num = int(n/batch_size)
logging.info("start training: TC")
for epoch in range(setting.TC):
    logging.debug('TC epoch: %d' % epoch)

    # create and use mask
    data_set_true = data_set.copy()
    mask = np.zeros((n, 1, setting.IMG_SIZE, setting.IMG_SIZE), dtype=np.float32)
    for i in range(1):
        # mask (32 ~ 128)
        logging.debug('create and use mask in %d/%d' % (i, n))
        mask_size = np.random.randint(32, 128)
        x_len = data_set[i].shape[1]
        y_len = data_set[i].shape[2]
        mask_begin_x = np.random.randint(0, x_len - mask_size)
        mask_begin_y = np.random.randint(0, y_len - mask_size)
        # create mask
        mask[i, 0, mask_begin_x:mask_begin_x+mask_size, mask_begin_y:mask_begin_y+mask_size] = \
            np.ones((mask_size, mask_size), dtype=np.float32)
        # use mask
        data_set[i, 0, mask_begin_x:mask_begin_x+mask_size, mask_begin_y:mask_begin_y+mask_size] = \
            np.zeros((mask_size, mask_size), dtype=np.float32)
        data_set_true *= mask
        # combine
        data_set = np.concatenate((data_set, mask), axis=1)
    logging.debug('create and use mask end')

    # get batch
    if setting.GPU:
        data_set = xp.array(data_set)
    sum_loss = 0
    shuffle_index = np.random.permutation(n)
    for start in range(0, n, batch_size):
        x_mask = data_set[shuffle_index[start:(start + batch_size)], ]
        x_original = Variable(data_set_true[shuffle_index[start:(start + batch_size)], ])
        mask = Variable(x_mask[:, 0, ])
        x = Variable(x_mask)
        print(x.shape)
        c = com(x)*mask
        com.cleargrads()

        loss = F.mean_squared_error(c, x_original)
        sum_loss += loss
        loss.backward()

    average_loss = sum_loss/batch_num
    logging.info("TC epoch %d: average loss = %f" % (epoch, average_loss))

    if epoch % 50 == 0:
        logging.info("save model")
        serializers.save_npz('model_com_%dx50.npz' % (epoch/50), com)

# TD
for epoch in range(setting.TD):
    pass

# TT
for epoch in range(setting.TC):
    pass

logging.info("end training !")

# Save Model
if setting.SAVE_MODEL:
    # serializers.save_npz('gpu_model.npz', model)
    com.to_cpu()
    dis.to_cpu()
    serializers.save_npz('model_gen.npz', com)
    serializers.save_npz('model_dis.npz', dis)
    logging.info('Model Saved')
