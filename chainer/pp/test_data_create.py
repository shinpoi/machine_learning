import numpy as np
from random import randint
import setting

num = setting.num
long = setting.long
dim = setting.dim
"""
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
"""

size = setting.size_2d
data_2d_x = np.zeros((num, 1, size, size), dtype=np.float32)
data_2d_y = np.zeros((num, 1, size, size), dtype=np.float32)
p_list = np.zeros((num, 4), dtype=np.float32)


def create_2d_path_x():
    img = np.zeros((1, size, size))
    x0 = randint(0, size-1)
    y0 = randint(0, size-1)
    xf = randint(0, size-1)
    yf = randint(0, size-1)
    img[0, x0, y0] = img[0, xf, yf] = 255
    p = np.array((x0, y0, xf, yf))
    return img, p


def create_2d_path_y(p):
    x0, y0, xf, yf = p
    img = np.zeros((1, size, size))
    step_x = xf - x0
    step_y = yf - y0
    # print("x0y0", x0, y0)
    # print("xfyf", xf, yf)
    # print("step", step_x, step_y)
    if step_x != 0 and step_y != 0:
        step = min(abs(step_x), abs(step_y))
        step_x = step_x / step
        step_y = step_y / step
    elif step_x == step_y == 0:
        step = 1
        step_x = step_y = 0
    else:
        step = max(abs(step_x), abs(step_y))
        step_x = step_x / step
        step_y = step_y / step

    # print("stpp", step_x, step_y)
    for i in range(step):
        # print(int(round(x0+i*step_x)), int(round(y0+i*step_y)))
        img[0, int(round(x0+i*step_x)), int(round(y0+i*step_y))] = 255
    img[0, xf, yf] = 255
    return img


for j in range(num):
    data_2d_x[j], p = create_2d_path_x()
    p_list[j] = p
    data_2d_y[j] = create_2d_path_y(p)
    if j % 1000 == 0:
        print('%d/%d' % (j, num))

np.save('data_artificial_x.npy', data_2d_x)
np.save('data_artificial_x.npy', data_2d_y)


