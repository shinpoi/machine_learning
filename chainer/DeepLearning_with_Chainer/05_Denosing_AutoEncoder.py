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
import matplotlib.pyplot as plt


########################
# Read data

# Data set from "https://osdn.net/projects/sfnet_irisdss/" IRIS.csv
url = 'IRIS.csv'

# read data
with open(url, 'r') as f:
    data_origin = csv.reader(f)
    data = []
    for i in data_origin:
        data.append(i[::2][:4])

data = np.array(data, dtype=np.float32)
x_train = data[:-25]
x_test = data[-25:]


########################
# Model


class DAE(Chain):
    def __init__(self):
        super(DAE, self).__init__(
            l1=L.Linear(4, 2),
            l2=L.Linear(2, 4),
        )

    def __call__(self, x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)

    def fwd(self, x):
        h1 = F.relu(self.l1(x))
        h2 = self.l2(h1)
        return h2

model = DAE()
optimizer = optimizers.SGD()
optimizer.setup(model)

################################
# Training
print("training start...")
st = time.clock()

n = 75
bs = 25
for i in range(6000):
    shuffle_index = np.random.permutation(n)
    for j in range(0, n, bs):
        x = Variable(x_train[shuffle_index[j:(j+bs) if (j+bs) < n else n]])
        model.cleargrads()
        loss = model(x)
        loss.backward()
        optimizer.update()

print("training time is: %f" % (time.clock() - st))
################################
# Evaluate
# 1. plot errors of Denosing-AutoEncoder model
plt.figure(1, figsize=(16, 9))
xt = Variable(x_test)
yt = F.relu(model.l1(xt))
xt_ = model.l2(yt)

errors = x_test**2 - xt_.data**2
er = np.zeros(len(errors), dtype=np.float32)
for i in range(len(errors)):
    er[i] = np.linalg.norm(errors[i])

plt.plot(range(len(er)), er, label="test data", color="red", linewidth=1)
plt.xlabel("number")
plt.ylabel("error")
plt.title("errors of Denosing-AutoEncoder model")
plt.legend()

# 2. plot data of model and original
plt.figure(2, figsize=(16, 9))
plt.plot(x_test[:, 0], x_test[:, 1], '.', label="origin data d1-d2", color="blue", linewidth=1)
plt.plot(xt_.data[:, 0], xt_.data[:, 1], '.', label="DAE data d1-d2", color="red", linewidth=1)

plt.plot(x_test[:, 2], x_test[:, 3], '.', label="origin data d3-d4", color="green", linewidth=1)
plt.plot(xt_.data[:, 2], xt_.data[:, 3], '.', label="DAE data d3-d4", color="purple", linewidth=1)

plt.xlabel("d1")
plt.ylabel("d2")
plt.title("data plot")
plt.legend()


plt.show()
