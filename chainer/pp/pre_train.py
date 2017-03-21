import numpy as np
import os
import sys
import math
from chainer import Chain, Variable, optimizers, serializers, cuda
import chainer.functions as F
import chainer.links as L
import setting

# Read data
# need shape of data: (batch, dim, height=1, width)
in_dim = 2*setting.dim
data = np.load('data_artificial.npy')
print("data's shape", data.shape)

x = np.zeros((data.shape[0], in_dim), dtype=np.float32)
x[:, :7] = data[:, :, 0, 0]
x[:, 7:] = data[:, :, 0, -1]
print("x's shape", x.shape)


# model
class GAN_HALF(Chain):
    def __init__(self):
        super(GAN_HALF, self).__init__(
            l=L.Linear(14, 1*2*512),  # 2
            dc1=L.Deconvolution2D(512, 256, (1, 5), stride=2),  # 7
            dc2=L.Deconvolution2D(256, 128, (1, 5), stride=2),  # 17
            dc3=L.Deconvolution2D(128, 64, (1, 5), stride=2),  # 37
            dc4=L.Deconvolution2D(64, 7, (1, 5), stride=2),  # 77
            bn0l=L.BatchNormalization(1*2*512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self, x, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l(x), test=test)), (x.data.shape[0], 512, 1, 2))
        # print('h1-end:', h.shape)
        h = F.relu(self.bn1(self.dc1(h), test=test))
        # print('h2-end:', h.shape)
        h = F.relu(self.bn2(self.dc2(h), test=test))
        # print('h3-end:', h.shape)
        h = F.relu(self.bn3(self.dc3(h), test=test))
        # print('h4-end:', h.shape)
        h = self.dc4(h)
        return h

if __name__ == "__main__":
    # model
    model = GAN_HALF()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)
    xp = cuda.cupy

    x_train = xp.array(x)
    y_train = xp.array(data)

    print("x_train's shape", x_train.shape)
    print("y_train's shape", y_train.shape)

    # train
    print('GPU used')

    n = data.shape[0]  # number of train-data
    bc = 500  # number of batch
    # train
    for j in range(1001):
        print('start loop %d' % j)

        # training
        sff_index = np.random.permutation(n)
        for i in range(0, n, bc):
            x = Variable(x_train[sff_index[i: (i + bc) if (i + bc) < n else n], ])
            y = Variable(y_train[sff_index[i: (i + bc) if (i + bc) < n else n], ])
            model.cleargrads()
            loss = model(x, y)
            loss.backward()
            optimizer.update()

    serializers.save_npz('cpu_model.npz', model)
    print('Model Saved')
