# - coding: utf-8 -*-
# python 3.4+

from chainer import Chain
import chainer.functions as F
import chainer.links as L
import setting

input_num = setting.RAND_IN_NUM
batch_num = setting.BATCH

"""
# original
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
        h = F.reshape(F.relu(self.bn0l(self.l0z(x))), (x.shape[0], 1024, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        h = F.relu(self.dc5(h))
        return h

    def __init__(self):
        super(Generator, self).__init__(
            l0z=L.Linear(input_num, 6 * 6 * 1024),
            dc1=L.Deconvolution2D(1024, 512, 4, stride=2, pad=1),  # 12x12x512
            dc11=L.Convolution2D(512, 512, 3, 1, 1),
            dc2=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),  # 24x24x256
            dc21=L.Convolution2D(256, 256, 3, 1, 1),
            dc3=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),  # 48x48x128
            dc31=L.Convolution2D(128, 128, 3, 1, 1),
            dc4=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),  # 96x96x64
            dc41=L.Convolution2D(64, 64, 3, 1, 1),
            dc5=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),  # 192x192x3
            dc51=L.Convolution2D(32, 3, 3, 1, 1),
            bn0l=L.BatchNormalization(6 * 6 * 1024),
            bn0=L.BatchNormalization(1024),
            bn1=L.BatchNormalization(512),
            bn11=L.BatchNormalization(512),
            bn2=L.BatchNormalization(256),
            bn21=L.BatchNormalization(256),
            bn3=L.BatchNormalization(128),
            bn31=L.BatchNormalization(128),
            bn4=L.BatchNormalization(64),
            bn41=L.BatchNormalization(64),
            bn5=L.BatchNormalization(32),
        )

    def __call__(self, x):
        h = F.reshape(F.relu(self.bn0l(self.l0z(x))), (x.shape[0], 1024, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn11(self.dc11(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn21(self.dc21(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn31(self.dc31(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        h = F.relu(self.bn41(self.dc41(h)))
        h = F.relu(self.bn5(self.dc5(h)))
        h = F.relu(self.dc51(h))
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
            # bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
            bn4=L.BatchNormalization(1024),
        )

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))
        h = F.relu(self.bn3(self.c3(h)))
        h = F.relu(self.bn4(self.c4(h)))
        h = self.l5l(h)
        return h
"""


#################################################################
# CIFAR-10
# RAND_IN_NUM = 400
class GeneratorCIFAR(Chain):

    def __init__(self):
        super(GeneratorCIFAR, self).__init__(
            l0z=L.Linear(input_num, 2 * 2 * 512),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),  # 4x4x256
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),  # 8x8x128
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),  # 16x16x64
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1),  # 32x32x3
            bn0l=L.BatchNormalization(2 * 2 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, x):
        h = F.reshape(F.relu(self.bn0l(self.l0z(x))), (x.shape[0], 512, 2, 2))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.dc4(h))
        return h


class DiscriminatorCIFAR(Chain):
    def __init__(self):
        super(DiscriminatorCIFAR, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1),  # 16x16x64
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1),  # 8x8x128
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1),  # 4x4x256
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1),  # 2x2x512
            l4l=L.Linear(2 * 2 * 512, 2),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))
        h = F.relu(self.bn3(self.c3(h)))
        h = self.l4l(h)
        return h


#################################################################
# Mnist
class GeneratorMnist(Chain):
    def __init__(self):
        super(GeneratorMnist, self).__init__(
            l0z=L.Linear(input_num, 4 * 4 * 256, nobias=True),
            dc1=L.Deconvolution2D(256, 128, 3, stride=2, pad=1, nobias=True),  # 7x7x256
            dc2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, nobias=True),  # 14x14x128
            dc3=L.Deconvolution2D(64, 1, 4, stride=2, pad=1, nobias=True),  # 28x28x1
            bn0l=L.BatchNormalization(4 * 4 * 256),
            bn0=L.BatchNormalization(256),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64),
        )

    def __call__(self, x):
        h = F.reshape(F.leaky_relu(self.bn0l(self.l0z(x))), (x.shape[0], 256, 4, 4))
        h = F.leaky_relu(self.bn1(self.dc1(h)))
        h = F.leaky_relu(self.bn2(self.dc2(h)))
        h = self.dc3(h)
        return h


class DiscriminatorMnist(Chain):
    def __init__(self):
        super(DiscriminatorMnist, self).__init__(
            c0=L.Convolution2D(1, 64, 4, stride=2, pad=1, nobias=True),  # 14x14x64
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1),  # 7x7x128
            c2=L.Convolution2D(128, 256, 3, stride=2, pad=1),  # 4x4x256
            l4l=L.Linear(4 * 4 * 256, 2),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(256),
        )

    def __call__(self, x):
        h = F.elu(self.bn1(self.c0(x)))
        h = F.elu(self.bn2(self.c1(h)))
        h = F.elu(self.bn3(self.c2(h)))
        h = self.l4l(h)
        return h

"""
# Face old
#################################################################
class GeneratorFace(Chain):
    def __init__(self):
        super(GeneratorFace, self).__init__(
            l0z=L.Linear(input_num, 6 * 6 * 512),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),  # 12
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),  # 24
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),   # 48
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1),     # 96
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, x):
        h = F.reshape(F.relu(self.bn0l(self.l0z(x))), (x.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        g = self.dc4(h)
        return g


class DiscriminatorFace(Chain):
    def __init__(self):
        super(DiscriminatorFace, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1),     # 48
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1),   # 24
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1),  # 12
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1),  # 6
            l4l=L.Linear(6 * 6 * 512, 2),
            # bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x):
        h = F.elu(self.c0(x))
        h = F.elu(self.bn1(self.c1(h)))
        h = F.elu(self.bn2(self.c2(h)))
        h = F.elu(self.bn3(self.c3(h)))
        d = self.l4l(h)
        return d
"""


class GeneratorFace(Chain):
    def __init__(self):
        super(GeneratorFace, self).__init__(
            l0z=L.Linear(input_num, 6 * 6 * 512, nobias=True),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1, nobias=True),  # 12
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),  # 24
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),   # 48
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1),     # 96
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, x):
        h = F.reshape(F.relu(self.bn0l(self.l0z(x))), (x.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        g = self.dc4(h)
        return g


class DiscriminatorFace(Chain):
    def __init__(self):
        super(DiscriminatorFace, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1, nobias=True),     # 48
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1),   # 24
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1),  # 12
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1),  # 6
            l4l=L.Linear(6 * 6 * 512, 2),
            # bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))
        h = F.relu(self.bn3(self.c3(h)))
        d = self.l4l(h)
        return d


class CompletionNet(Chain):
    def __init__(self):
        super(CompletionNet, self).__init__(
            c0a=L.Convolution2D(2, 64, 5, stride=1, pad=2),
            c1a=L.Convolution2D(64, 128, 3, 2, 1),      # 1/2
            c1b=L.Convolution2D(128, 128, 1, 1, 1),
            c2a=L.Convolution2D(128, 256, 3, 2, 1),     # 1/4
            c2b=L.Convolution2D(256, 256, 3, 1, 1),
            c2c=L.Convolution2D(256, 256, 3, 1, 1),
            dlc2a=L.DilatedConvolution2D(256, 256, 3, stride=1, pad=2, dilate=2),
            dlc2b=L.DilatedConvolution2D(256, 256, 3, stride=1, pad=4, dilate=4),
            dlc2c=L.DilatedConvolution2D(256, 256, 3, stride=1, pad=8, dilate=8),
            dlc2d=L.DilatedConvolution2D(256, 256, 3, stride=1, pad=16, dilate=16),
            c2d=L.Convolution2D(256, 256, 3, 1, 1),
            c2e=L.Convolution2D(256, 256, 3, 1, 1),
            dc3a=L.Deconvolution2D(256, 128, 4, 2, 1),  # 1/2
            c3a=L.Convolution2D(128, 128, 3, 1, 1),
            dc4a=L.Deconvolution2D(128, 64, 4, 2, 1),  # 1
            c4a=L.Convolution2D(64, 32, 3, 1, 1),
            c4b=L.Convolution2D(32, 1, 3, 1, 1),

            bn0a=L.BatchNormalization(64),
            bn1a=L.BatchNormalization(128),
            bn1b=L.BatchNormalization(128),
            bn2a=L.BatchNormalization(256),
            bn2b=L.BatchNormalization(256),
            bn2c=L.BatchNormalization(256),
            bn2da=L.BatchNormalization(256),
            bn2db=L.BatchNormalization(256),
            bn2dc=L.BatchNormalization(256),
            bn2dd=L.BatchNormalization(256),
            bn2d=L.BatchNormalization(256),
            bn2e=L.BatchNormalization(256),
            bn3a=L.BatchNormalization(128),
            bn3b=L.BatchNormalization(128),
            bn4a=L.BatchNormalization(64),
            bn4b=L.BatchNormalization(32),
        )

    def __call__(self, x):
        h = F.relu(self.bn0a(self.c0a(x)))
        h = F.relu(self.bn1a(self.c1a(h)))
        h = F.relu(self.bn1b(self.c1b(h)))
        h = F.relu(self.bn2a(self.c2a(h)))
        h = F.relu(self.bn2b(self.c2b(h)))
        h = F.relu(self.bn2c(self.c2c(h)))
        h = F.relu(self.bn2da(self.dlc2a(h)))
        h = F.relu(self.bn2db(self.dlc2b(h)))
        h = F.relu(self.bn2dc(self.dlc2c(h)))
        h = F.relu(self.bn2dd(self.dlc2d(h)))
        h = F.relu(self.bn2d(self.c2d(h)))
        h = F.relu(self.bn2e(self.c2e(h)))
        h = F.relu(self.bn3a(self.dc3a(h)))
        h = F.relu(self.bn3b(self.c3b(h)))
        h = F.relu(self.bn4a(self.dc4a(h)))
        h = F.sigmoid(self.bn4b(self.c4b(h)))
        return h


class LGDiscriminator(Chain):
    def __init__(self):
        super(LGDiscriminator, self).__init__(
            Lc0=L.Convolution2D(1, 64, 5, 2, 2),     # -> 64x64
            Lc1=L.Convolution2D(64, 128, 5, 2, 2),   # -> 32x32
            Lc2=L.Convolution2D(128, 256, 5, 2, 2),   # -> 16x16
            Lc3=L.Convolution2D(256, 512, 5, 2, 2),   # -> 8x8
            Lc4=L.Convolution2D(512, 512, 5, 2, 2),   # -> 4x4
            Lfc=L.Linear(2 * 2 * 512, 1024),

            Lbn0=L.BatchNormalization(64),
            Lbn1=L.BatchNormalization(128),
            Lbn2=L.BatchNormalization(256),
            Lbn3=L.BatchNormalization(512),
            Lbn4=L.BatchNormalization(512),

            Gc0=L.Convolution2D(1, 64, 5, 2, 2),  # -> 128x128
            Gc1=L.Convolution2D(64, 128, 5, 2, 2),  # -> 64x64
            Gc2=L.Convolution2D(128, 256, 5, 2, 2),  # -> 32x32
            Gc3=L.Convolution2D(256, 512, 5, 2, 2),  # -> 16x16
            Gc4=L.Convolution2D(512, 512, 5, 2, 2),  # -> 8x8
            Gc5=L.Convolution2D(512, 512, 5, 2, 2),  # -> 4x4
            Gfc=L.Linear(2 * 2 * 512, 1024),

            Gbn0=L.BatchNormalization(64),
            Gbn1=L.BatchNormalization(128),
            Gbn2=L.BatchNormalization(256),
            Gbn3=L.BatchNormalization(512),
            Gbn4=L.BatchNormalization(512),
            Gbn5=L.BatchNormalization(512),

            fc=L.Linear(2 * 2 * 512 * 2, 2)
        )

    def __call__(self, l, g):
        hl = F.relu(self.Lbn0(self.Lc0(l)))
        hl = F.relu(self.Lbn1(self.Lc1(hl)))
        hl = F.relu(self.Lbn2(self.Lc2(hl)))
        hl = F.relu(self.Lbn3(self.Lc3(hl)))
        hl = F.relu(self.Lbn4(self.Lc4(hl)))
        hl = F.relu(self.Lfc(hl))

        hg = F.relu(self.Gbn0(self.Gc0(l)))
        hg = F.relu(self.Gbn1(self.Gc1(hg)))
        hg = F.relu(self.Gbn2(self.Gc2(hg)))
        hg = F.relu(self.Gbn3(self.Gc3(hg)))
        hg = F.relu(self.Gbn4(self.Gc4(hg)))
        hg = F.relu(self.Gbn5(self.Gc5(hg)))
        hg = F.relu(self.Lfc(hg))

        h = F.sigmoid(self.fc(F.concat((hl, hg))))
        return h





