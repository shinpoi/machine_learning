# - coding: utf-8 -*-
# python 3.4+

from chainer import Variable, serializers, using_config, no_backprop_mode
import model
import numpy as np
import matplotlib.pyplot as plt

MIN_START = 0.08
MAX_START = 0.15
START = MAX_START - MIN_START

MIN_END = 0.85
MAX_END = 0.95
END = MAX_END - MIN_END

OBS_MIN = 0.3
OBS_MAX = 0.7
OBS = OBS_MAX - OBS_MIN

n = 10
TRAIN = False

print('load model...')
generator = model.GeneratorPath()
# serializers.load_npz('model_gen_6x100.npz', generator)

x = np.zeros((n, 24), dtype=np.float32)
for i in range(n):
    for j in range(8):
        x[i, j] = np.random.random()*OBS + OBS_MIN
    x[i, 8:10] = [np.random.random()*START + MIN_START, np.random.random()*START + MIN_START]
    x[i, 10:12] = [np.random.random()*END + MIN_END, np.random.random()*END + MIN_END]
    for j in range(12):
        x[i, 12+j] = np.random.random()

vx = Variable(np.array(x, dtype=np.float32))
y = np.zeros((n, 1, 2, 45), dtype=np.float32)

# use train data
if TRAIN:
    print("use train data")
    data_head = np.array(np.load('/home/shin-u16/document/dataset/rrt/rrtstar_head.npy'),
                         dtype=np.float32)[50:50+n]
    x[:, :12] = data_head


model = 160
for i in range(model):
    serializers.load_npz('./model_com_%dx50.npz' % i, generator)
    for sample in y:
        with using_config('train', False):
            with no_backprop_mode():
                y = generator(vx).data

    np.save("./path/path_%dx50.npy" % i, y)

print('plot start')
for i in range(model):
    data = np.load("./path/path_%dx50.npy" % i)
    for j in range(n):
        x_ = data[j, 0, 0]
        y_ = data[j, 0, 1]
        plt.figure(figsize=(12, 9))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        area = np.pi * 3000
        for k in range(4):
            plt.scatter(x[j, k*2], x[j, k*2+1], s=area, color='blue', alpha=0.5)
        plt.scatter(x[j, 8], x[j, 9], color='red', alpha=0.5)
        plt.scatter(x[j, 10], x[j, 11], color='red', alpha=0.5)
        plt.plot(x_, y_)
        plt.savefig('./path/res/%d_%d.png' % (i, j), dpi=75)
        plt.close()

"""
# sample by rrt
data_path = np.array(np.load('/home/shin-u16/document/dataset/rrt/rrtstar_path.npy'), dtype=np.float32)[:10]
data_head = np.array(np.load('/home/shin-u16/document/dataset/rrt/rrtstar_head.npy'), dtype=np.float32)[:10]
for i in range(10):
    x_ = data_path[i, :, 0]
    y_ = data_path[i, :, 1]
    plt.figure(figsize=(12, 9))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    area = np.pi * 3000
    for k in range(4):
        plt.scatter(data_head[i, k * 2], data_head[i, k * 2 + 1], s=area, color='orange', alpha=0.5)
    plt.scatter(data_head[i, 8], data_head[i, 9], color='green', alpha=0.5)
    plt.scatter(data_head[i, 10], data_head[i, 11], color='green', alpha=0.5)
    plt.plot(x_, y_)
    plt.savefig('./path/res/sample_%d.png' % i, dpi=75)
    plt.close()
"""


