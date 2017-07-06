# - coding: utf-8 -*-
# python 3.4+

from chainer import Chain
import chainer.functions as F
import chainer.links as L
import setting

input_num = setting.RAND_IN_NUM
batch_num = setting.BATCH


class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z=L.Linear(input_num, 6 * 6 * 1024),
            dc1=L.Deconvolution2D(1024, 512, 4, stride=2, pad=1),  # 12x12x512
            dc2=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),  # 24x24x256
            dc3=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),  # 48x48x128
            dc4=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),  # 96x96x64
            dc5=L.Deconvolution2D(64, 3, 4, stride=2, pad=1),  # 192x192x3
            bn0l=L.BatchNormalization(6 * 6 * 1024),
            bn0=L.BatchNormalization(1024),
            bn1=L.BatchNormalization(512),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(64),
        )

    def __call__(self, x):
        h = F.reshape(F.relu(self.bn0l(self.l0z(x))), (batch_num, 1024, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        h = self.dc5(h)
        return h


class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1),  # 96x96x64
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1),  # 48x48x128
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1),  # 24x24x256
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1),  # 12x12x512
            c4=L.Convolution2D(512, 1024, 4, stride=2, pad=1),  # 6x6x1024
            l5l=L.Linear(6 * 6 * 1024, 2),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
            bn4=L.BatchNormalization(1024),
        )

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))  # no bn because images from generator will katayotteru?
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))
        h = F.relu(self.bn3(self.c3(h)))
        h = F.relu(self.bn4(self.c4(h)))
        h = self.l5l(h)
        return h