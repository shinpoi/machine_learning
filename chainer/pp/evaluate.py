from pre_train import GAN_HALF
from chainer import Variable, serializers
import setting
import matplotlib.pyplot as plt
import numpy as np

dim = setting.dim
long = setting.long

# load model
model = GAN_HALF()
serializers.load_npz('cpu_model.npz', model)

# create data-x (1, 14)
start = np.zeros(dim, dtype=np.float32)
end = np.zeros(dim, dtype=np.float32)
for i in range(dim):
    start[i] = np.random.randint(0, 361)
    end[i] = np.random.randint(0, 361)

x = np.zeros((1, 2*dim), dtype=np.float32)
x[:, :7] = start
x[:, 7:] = end

# create data-y (long, 7)
yt = np.zeros((long, 7))
yt[0] = start
yt[-1] = end
approach = (end - start)/(len(yt)-1)
for i in range(1, len(yt)-1):
    yt[i] = yt[i-1] + approach

print('yt:', yt[:, 1])

# predict data-y
xv = Variable(x)
yv = model.fwd(xv)
yp = yv.data[0, :, 0, :]  # (1,7,1,long) -> (7,long)

print('yp:', yp[1, :])

# evaluate
"""
for i in range(len(yp)):
    plt.figure(i, figsize=(4, 3))
    plt.plot(range(len(yt)), yt[:, i], label="True data | p" + str(i), color="blue", linewidth=1)
    plt.plot(range(len(yt)), yp[i, :], label="Predict data | p" + str(i), color="gray", linewidth=1)
    plt.title("data-p" + str(i))
    plt.xlabel("times")
    plt.ylabel("angle")
    plt.legend()
"""

plt.figure(1, figsize=(16, 9))
plt.plot(range(len(yt)), yt[:, 1] - yp[1, :], label="error p1", color="red", linewidth=1)
plt.plot(range(len(yt)), yt[:, 2] - yp[2, :], label="error p2", color="blue", linewidth=1)
plt.plot(range(len(yt)), yt[:, 3] - yp[3, :], label="error p3", color="green", linewidth=1)
plt.plot(range(len(yt)), yt[:, 4] - yp[4, :], label="error p4", color="gray", linewidth=1)
plt.plot(range(len(yt)), yt[:, 5] - yp[5, :], label="error p5", color="yellow", linewidth=1)
plt.plot(range(len(yt)), yt[:, 6] - yp[6, :], label="error p5", color="black", linewidth=1)
plt.title("data-p")
plt.xlabel("times")
plt.ylabel("angle")
plt.legend()
plt.show()
