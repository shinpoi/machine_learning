# - coding: utf-8 -*-
# python 3.4+

import numpy as np
from chainer import Variable, optimizers, optimizer, serializers, cuda
import chainer.functions as F
import logging
import setting
import model


##################
#  import true image set (192x192)
logging.info("start import dataset...")

batch_num = setting.BATCH
input_num = setting.RAND_IN_NUM

data_set = np.array(np.load('dataset.npy'), dtype=np.float32)
n = len(data_set)

##################
# set model

# load model
logging.info("start load model...")

gen = model.Generator()
dis = model.Discriminator()

opt_gen = optimizers.Adam(alpha=setting.ADAM_RATE)
opt_dis = optimizers.Adam(alpha=setting.ADAM_RATE)
opt_gen.setup(gen)
opt_dis.setup(dis)
if setting.WeightDecay:
    opt_gen.add_hook(optimizer.WeightDecay(setting.WeightDecay))
    opt_dis.add_hook(optimizer.WeightDecay(setting.WeightDecay))

# Use GPU
if setting.GPU:
    gpu_device = 0
    cuda.get_device_from_id(gpu_device).use()
    gen.to_gpu(gpu_device)
    dis.to_gpu(gpu_device)
    xp = cuda.cupy
    data_set = xp.array(data_set)
    logging.info('GPU used')

##################
#  train
# True = 1, Fake = 0
# (nx1)x --> (3x192x192)t --> (2x1)y

logging.info("start training ...")
for epoch in range(setting.EPOCH):
    logging.debug('start epoch: %d' % epoch)
    shuffle_index = np.random.permutation(n)
    # for every batch:
    for start in range(0, n, batch_num):
        if (start + batch_num) > n:
            continue

        x = xp.random.uniform(-1, 1, (batch_num, input_num))
        x = Variable(xp.array(x, dtype=np.float32))
        # print('x.shape:', x.shape)

        t_gen = gen(x)
        t_true = Variable(data_set[shuffle_index[start:(start+batch_num)], ])
        # print('t_true.shape:', t_true.shape)

        y_gen = dis(t_gen)
        y_true = dis(t_true)

        loss_gen = F.softmax_cross_entropy(y_gen, Variable(xp.ones(batch_num, dtype=np.int32)))
        loss_dis = F.softmax_cross_entropy(y_gen, Variable(xp.zeros(batch_num, dtype=np.int32)))
        loss_dis += F.softmax_cross_entropy(y_true, Variable(xp.ones(batch_num, dtype=np.int32)))

        gen.cleargrads()
        loss_gen.backward()
        opt_gen.update()

        dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        logging.debug("loss of generator: %s" % str(loss_gen))
        logging.debug("loss of discriminator: %s" % str(loss_dis))

    # evaluate
        # pass

logging.info("end training !")

# Save Model
if setting.SAVE_MODEL:
    # serializers.save_npz('gpu_model.npz', model)
    gen.to_cpu()
    dis.to_cpu()
    serializers.save_npz('model_gen.npz', gen)
    serializers.save_npz('model_dis.npz', dis)
    logging.info('Model Saved')
