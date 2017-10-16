# -*- coding: utf-8 -*-
# Python3

import numpy as np
import logging
import math
from chainer import cuda, Variable, optimizers, serializers, Chain, using_config, no_backprop_mode
import chainer.functions as F
import chainer.links as L


# parameter
logging.basicConfig(level="DEBUG",
                    format='[%(levelname)s]   \t%(message)s')
vocab = {}
GPU = False

if GPU:
    xp = cuda.cupy
    logging.info('GPU used')
else:
    xp = np
    logging.info('CPU used')


# RNN model
class MyRNN(Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed=L.EmbedID(v, k),
            H=L.Linear(k, k),
            W=L.Linear(k, v),
        )

    def __call__(self, s):
        accum_loss = 0
        v, k = self.embed.W.data.shape
        h = Variable(xp.zeros((1, k), dtype=np.float32))
        for i in range(len(s)):
            try:
                next_word_id = s[i+1]
            except IndexError:
                next_word_id = eos_id
            tx = Variable(xp.array([next_word_id], dtype=np.int32))
            x_k = self.embed(Variable(xp.array([s[i]], dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss += loss
        return accum_loss

    def output(self, s):
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1, k), dtype=np.float32))
        sum_ = 0.0
        out = [s[0], ]
        for i in range(1, len(s)):
            w1, w2 = s[i-1], s[i]
            x_k = self.embed(Variable(np.array([w1], dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            yv = F.softmax(self.W(h))
            pi = yv.data[0][w2]
            sum_ -= math.log(pi, 2)
            out.append(np.argmax(yv.data))
        return sum_, out


def load_data(filename):
    global vocab
    with open(filename) as f:
        words = f.read().replace('\n', '<eos>').strip().split()
    data_set = np.ndarray((len(words), ), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        data_set[i] = vocab[word]
    # list --> Arr[<NO. of word>, ..., ..., ..., .....]
    return data_set


# load data
logging.info('Load training data ...')
train_data = load_data("src/ptb.train.txt")
eos_id = vocab['<eos>']

sentence = [[], ]
n = 0
for i in range(len(train_data)):
    id_ = train_data[i]
    sentence[n].append(id_)
    if id_ == eos_id:
        n += 1
        sentence.append([])

sentence = sentence[:-1]
n += -1
print("get %d sentence" % n)

# load model
demb = 100
model = MyRNN(len(vocab), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

# Use GPU
if GPU:
    gpu_device = 0
    cuda.get_device(gpu_device).use()
    # model.to_gpu(gpu_device)
    logging.info('GPU model added')

# training
logging.info('training start')
for epoch in range(50):
    #logging.info('Epoch %d' % epoch)
    print('Epoch %d' % epoch)
    for i in range(n):
        if i % 500 == 0:
            #logging.debug('sentence %d / %d' % (i, n))
            print('sentence %d / %d' % (i, n))
        model.cleargrads()
        loss = model(xp.array(sentence[i]))
        loss.backward()
        optimizer.update()
        
serializers.save_npz("cpu_model_07.npz", model)
logging.info('Training finished')

"""
# eval
# load test data
logging.info('Load test data ...')
test_data = load_data("src/ptb.test.txt")
sentence = [[], ]
n = 0
for i in range(len(test_data)):
    id_ = test_data[i]
    sentence[n].append(id_)
    if id_ == eos_id:
        n += 1
        sentence.append([])

sentence = sentence[:-1]
n += -1
print("get %d sentence" % n)

# create inv-vocab || {num: word}
inv_vocab = {}
for word in vocab.keys():
    inv_vocab[vocab[word]] = word

# load model
demb = 100
model = MyRNN(len(vocab), demb)
serializers.load_npz("cpu_model_07.npz", model)

# eval
print("put any key to next ...")

for x in sentence:
    with no_backprop_mode():
        with using_config('train', False):
            loss, y = model.output(xp.array(x))
    print("ori:", [inv_vocab[i] for i in x])
    print("gen:", [inv_vocab[i] for i in y])
    print("loss: %f \n" % loss)
    msg = input("Enter to next ... \n")
"""
