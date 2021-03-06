# -*- coding: utf-8 -*-
# Python2.7 & Python3.5
# TensorFlow r0.11

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time


time_start = time.time()
############################

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Pooling DataSet
def func_reduce(images, rank, pool):
    li = []
    for i in range(rank)[::pool]:
        for j in range(rank)[::pool]:
            li.append(28 * i + j)
    reduce_list = []
    for img in images:
        temp_list = []
        for n in li:
            temp_list.append((img[n] + img[n+1] + img[n+rank] + img[n+1+rank]) / pool**2)
        reduce_list.append(temp_list)
    return np.array(reduce_list)

# Reduce DataSet
rank = 28
pool = 2

scale_train = 50
train_images = func_reduce(images=mnist.train.images[::scale_train], rank=rank, pool=pool)
train_labels = mnist.train.labels[::scale_train]

scale_test = 10
test_images = func_reduce(images=mnist.test.images[::scale_test], rank=rank, pool=pool)
test_labels = mnist.test.labels[::scale_test]

print("Reduce DataSet : Finished")
#############################
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 196])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer
W_conv1 = weight_variable([4, 4, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 14, 14, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([4, 4, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

print("Train : Start")
# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

loop = 0
for i in range(10000):
    # batch = mnist.train.next_batch(50)
    z = i - loop*22
    if z*50 + 50 > 1100:
        loop += 1
        z = i - loop*22

    batch_images = train_images[z*50:z*50+50]
    batch_labels = train_labels[z*50:z*50+50]
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))
print("time is: %d" % (time.time() - time_start))
