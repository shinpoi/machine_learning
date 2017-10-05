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
log_name = setting.LOG_DIR + 'path_train_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
logging.basicConfig(filename=log_name)

#  import true image set (192x192)
logging.info("start import dataset...")

batch_size = setting.BATCH

data_path = np.array(np.load('/home/shin-u16/document/dataset/rrt/rrtstar_path.npy'), dtype=np.float32)[:50000]
data_head = np.array(np.load('/home/shin-u16/document/dataset/rrt/rrtstar_head.npy'), dtype=np.float32)[:50000]

n = len(data_path)
logging.info("data_path.shape: %s, data_head.shape: %s" % (str(data_path.shape), str(data_head.shape)))

##################
# set model

# create x&y
x = np.zeros((n, 24), dtype=np.float32)
x[:, :12] = data_head
# x[:, 0, 12:] = np.random.uniform(0, 1, (n, 1, 12))

y = data_path.transpose((0, 2, 1)).reshape((n, 1, 2, 45))

del data_path
del data_head

logging.info("x.shape: %s, y.shape: %s" % (str(x.shape), str(y.shape)))

# load model
logging.info("start load model...")

gen = model.GeneratorPath()
# dis = model.LGDiscriminator()

opt_gen = optimizers.Adam(alpha=setting.ADAM_RATE)
opt_gen.setup(gen)
# opt_dis = optimizers.Adam(alpha=setting.ADAM_RATE)
# opt_dis.setup(dis)
if setting.WeightDecay:
    opt_gen.add_hook(optimizer.WeightDecay(setting.WeightDecay))
    # opt_dis.add_hook(optimizer.WeightDecay(setting.WeightDecay))

# Use GPU
if setting.GPU:
    gpu_device = 0
    cuda.get_device_from_id(gpu_device).use()
    gen.to_gpu(gpu_device)
    # dis.to_gpu(gpu_device)
    xp = cuda.cupy
    x = xp.array(x)
    y = xp.array(y)
    logging.info('GPU used')

##################
#  train
# True = 1, Fake = 0
batch_num = int(n/batch_size)
logging.info("start training: TC")
for epoch in range(setting.TC):
    logging.debug('TC epoch: %d' % epoch)

    # random para
    x[:, 12:] = xp.random.uniform(0, 1, (n, 12))

    # get batch
    sum_loss = 0
    shuffle_index = np.random.permutation(n)
    for start in range(0, n, batch_size):
        # print(start)
        x_batch = Variable(x[shuffle_index[start:(start + batch_size)], ])
        y_batch = Variable(y[shuffle_index[start:(start + batch_size)], ])

        res = gen(x_batch)
        gen.cleargrads()
        loss = F.mean_squared_error(res, y_batch)
        loss.backward()
        opt_gen.update()
        sum_loss += loss.data

    average_loss = sum_loss/batch_num
    logging.info("TC epoch %d: average loss = %f" % (epoch, average_loss))

    if epoch % 50 == 0:
        logging.info("save model")
        serializers.save_npz('model_com_%dx50.npz' % (epoch/50), gen)

# TD
for epoch in range(setting.TD):
    pass

# TT
for epoch in range(setting.TC):
    pass

logging.info("end training !")

# Save Model
if setting.SAVE_MODEL:
    serializers.save_npz('model_gen.npz', gen)
    # serializers.save_npz('model_dis.npz', dis)
    logging.info('Model Saved')
