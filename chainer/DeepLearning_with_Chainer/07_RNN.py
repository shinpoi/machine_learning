# -*- coding: utf-8 -*-
# Python3

import numpy as np
import logging
import math
from chainer import cuda, Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L


# parameter
logging.basicConfig(level="DEBUG",
                    format='[%(levelname)s]   \t%(message)s')
vocab = {}
GPU = True

if GPU:
    xp = cuda.cupy
    logging.info('GPU used')
else:
    xp = np
    logging.info('CPU used')


# read data
def load_data(filename):
    global vocab
    with open(filename) as f:
        words = f.read().replace('\n', '<eos>').strip().split()
    data_set = np.ndarray((len(words), ), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        data_set[i] = vocab[word]
    return data_set


# RNN model
class MyRNN(Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed=L.EmbedID(v, k),
            H=L.Linear(k, k),
            W=L.Linear(k, v)
        )

    def __call__(self, s):
        accum_loss = None
        v, k = self.embed.W.data.shape
        h = Variable(xp.zeros((1, k), dtype=np.float32))
        for i in range(len(s)):
            next_word_id = eos_id if (i == len(s) - 1) else s[i+1]
            tx = Variable(xp.array([next_word_id], dtype=np.int32))
            x_k = self.embed(Variable(xp.array([s[i]], dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    def output(self, s):
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros(1, k), dtype=np.float32)
        sum = 0.0
        for i in range(1, len(s)):
            w1, w2 = s[i-1], s[i]
            x_k = self.embed(Variable(np.array([w1]), dtype=np.int32))
            h = F.tanh(x_k + self.H(h))
            yv = F.softmax(self.W(h))
            pi = yv.data[0][w2]
            print("yv.data.shape", yv.data[0].shape)
            print("yv.data[0]", yv.data[0])
            sum -= math.log(pi, 2)
        return sum


# load data
logging.info('Load training data ...')
train_data = load_data("src/ptb.train.txt")
eos_id = vocab['<eos>']

# load model
demb = 100
model = MyRNN(len(vocab), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

# Use GPU
if GPU:
    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)
    logging.info('GPU model added')


# training
logging.info('Training start')
for epoch in range(5):
    logging.info('Gpoch %d' % epoch)
    s = []
    num_words = len(train_data)
    for i in range(num_words):
        if i % 10000 == 0:
            logging.debug('word %d / %d' % (i, num_words))
        id_ = train_data[i]
        s.append(id_)
        if id_ == eos_id:
            model.cleargrads()
            loss = model(np.array(s))
            loss.backward()
            optimizer.update()
            s = []
    serializers.save_npz("cpu_model_07.npz", model)

logging.info('Training finished')

# eval
# nu = model.output(['He', 'is', 'a', 'new'])
# nu_1 = int(nu) - 1
# nu0 = int(nu)
# nu2 = int(nu) + 1
# print(vocab[nu_1], vocab[nu0], vocab[nu2])
