# -*- coding: utf-8 -*-
# TensorFlow r0.11
# https://www.tensorflow.org/tutorials/tflearn/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv

# Data set
url = '/home/shin-u16/document/tensorflow_learn/IRIS.csv'

with open(url, 'r') as f:
    a = csv.reader(f)
    b = []
    for i in a:
        b.append(i)

fo = {'setosa': 0, 'versicolor': 2, 'virginica': 1}

for i in b:
    i[-1] = fo[i[-1]]

tr = b[:80]
ts = b[80:]

tr = np.array(tr, dtype='float32')
ts = np.array(ts, dtype='float32')

print(tr.shape, ts.shape)

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")