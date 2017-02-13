# - coding: utf-8 -*-
# Python3.5 & Chainer 1.20.0.1

import numpy as np
from chainer import Variable
from chainer import optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import csv
import time

################################
st = time.clock()
# Data set from "https://osdn.net/projects/sfnet_irisdss/" IRIS.csv
url = 'IRIS.csv'

# read data
with open(url, 'r') as f:
    data_origin = csv.reader(f)
    data = []
    for i in data_origin:
        data.append(i[::2])

# target: string -> number
fo = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

for i in data:
    i[-1] = fo[i[-1]]

# data -> feature, target
data = np.array(data, dtype='float32')

data_x = data[:, :4]
data_y = data[:, -1].astype(np.int32)

# target -> one-hot array
d = np.zeros([len(data_y), 3])
for i in range(len(data_y)):
    d[i][data_y[i]] = 1
d = d.astype(np.float32)

# data -> train & test
x_train = data_x[::2]
y_train = data_y[::2]   # one-hot
# y_train = d[::2]  # single-value

x_test = data_x[1::2]
y_test = data_y[1::2]

print("read time is: %f" % (time.clock() - st))
st = time.clock()
################################
# Model


class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


# don't use one-hot array
class IrisChainSofmax(Chain):
    def __init__(self):
        super(IrisChainSofmax, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


class IrisChainLogistic(Chain):
    def __init__(self):
        super(IrisChainLogistic, self).__init__(
            l1=L.Linear(4, 3),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x):
        h1 = F.softmax(self.l1(x))
        return h1


model = IrisChainSofmax()
optimizer = optimizers.SGD()
optimizer.setup(model)

################################
# Training
"""
for i in range(1000):
    x = Variable(x_train)
    y = Variable(y_train)
    model.cleargrads()
    loss = model(x, y)
    loss.backward()
    optimizer.update()
"""

# Training by mini-batch
n = 50
bs = 25
for j in range(5000):
    sff_index = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(x_train[sff_index[i: (i + bs) if (i + bs) < n else n]])
        y = Variable(y_train[sff_index[i: (i + bs) if (i + bs) < n else n]])
        model.cleargrads()
        loss = model(x, y)
        loss.backward()
        optimizer.update()

print("training time is: %f" % (time.clock() - st))
st = time.clock()
################################
# Evaluate

xt = Variable(x_test, volatile='on')
yt = model.fwd(xt)
ans = yt.data
nrow, ncol = ans.shape
ok = 0

for i in range(nrow):
    cls = np.argmax(ans[i])
    if cls == y_test[i]:
        ok += 1

print("evaluate time is: %f" % (time.clock() - st))
st = time.clock()

print("\n")
print("accuracy: %d/%d = %f" % (ok, nrow, ok*1.0/nrow))
