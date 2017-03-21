import numpy as np
from random import randint
import setting

num = setting.num
long = setting.long
dim = setting.dim

data = np.zeros((num, dim, 1, long), dtype=np.float32)


def create_path():
    start = np.zeros((dim, 1), dtype=np.float32)
    end = np.zeros((dim, 1), dtype=np.float32)
    path = np.zeros((dim, 1, long), dtype=np.float32)

    for i in range(dim):
        start[i] = randint(0, 360)
        end[i] = randint(0, 360)

    path[:, :, 0] = start
    path[:, :, -1] = end
    approach = (end - start)/(long-1)

    for i in range(1, long-1):
        path[:, :, i] = path[:, :, i-1] + approach

    return path

for j in range(num):
    data[j] = create_path()
    if j % 1000 == 0:
        print('%d/%d' % (j, num))

np.save('data_artificial.npy', data)
