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
log_name = setting.LOG_DIR + 'train_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
logging.basicConfig(filename=log_name)

#  import true image set (192x192)
logging.info("start import dataset...")

batch_num = setting.BATCH
input_num = setting.RAND_IN_NUM

data_set = np.array(np.load('./data_face.npy'), dtype=np.float32)
data_set = data_set[:len(data_set) - (len(data_set) % batch_num)]
data_set /= 255.0
n = len(data_set)
logging.info("get %d true data for training..." % n)

##################
# set model

# load model
logging.info("start load model...")

gen = model.GeneratorFace()
dis = model.DiscriminatorFace()

opt_gen = optimizers.Adam(alpha=setting.ADAM_RATE)
opt_gen.setup(gen)
opt_dis = optimizers.Adam(alpha=setting.ADAM_RATE)
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
    sum_loss_gen = np.float32(0)
    sum_loss_dis = np.float32(0)

    if epoch % 5 == 0:
        tp = tn = 0

    # for every batch:
    for start in range(0, n, batch_num):
        """
        if (start + batch_num) > n:
            continue
        """

        x = xp.random.uniform(0, 1, (batch_num, input_num))
        x = Variable(xp.array(x, dtype=np.float32))
        # print('x.shape:', x.shape)

        t_gen = gen(x)
        t_true = Variable(data_set[shuffle_index[start:(start+batch_num)], ])
        # print('t_true.shape:', t_true.shape)

        y_gen = dis(t_gen)
        y_true = dis(t_true)

        gen.cleargrads()
        dis.cleargrads()

        loss_gen = F.softmax_cross_entropy(y_gen, Variable(xp.ones(batch_num, dtype=np.int32)))
        loss_gen.backward()
        opt_gen.update()

        loss_dis = F.softmax_cross_entropy(y_gen, Variable(xp.zeros(batch_num, dtype=np.int32)))
        loss_dis += F.softmax_cross_entropy(y_true, Variable(xp.ones(batch_num, dtype=np.int32)))
        loss_dis.backward()
        opt_dis.update()

        sum_loss_gen += loss_gen.data.get()
        sum_loss_dis += loss_dis.data.get()

        # evaluate
        if epoch % 5 == 0:
            y_true_data = y_true.data
            y_gen_data = y_gen.data
            for i in range(batch_num):
                # print(d_true[i])
                if y_true_data[i][1] > y_true_data[i][0]:
                    tp += 1
                # tp += d_true[i].argmax()
                # print(d_gen[i])
                if y_gen_data[i][0] > y_gen_data[i][1]:
                    tn += 1

    logging.debug("loss of generator: %f" % sum_loss_gen)
    logging.debug("loss of discriminator: %f" % sum_loss_dis)

    # evaluate
    if epoch % 5 == 0:
        logging.info("acc of dis-true: %d/%d = %f" % (tp, n, tp/float(n)))
        logging.info("acc of dis-fake: %d/%d = %f" % (tn, n, tn/float(n)))

    # middle save
    if setting.SAVE_MODEL:
        if epoch % 50 == 0:
            serializers.save_npz('model_gen_%sx50.npz' % str(int(epoch/50)), gen)
            logging.info('Model Saved')

logging.info("end training !")

# Save Model
if setting.SAVE_MODEL:
    # serializers.save_npz('gpu_model.npz', model)
    gen.to_cpu()
    dis.to_cpu()
    serializers.save_npz('model_gen.npz', gen)
    serializers.save_npz('model_dis.npz', dis)
    logging.info('Model Saved')
