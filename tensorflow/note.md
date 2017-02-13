tensorflow learning note
=================================
0.environment
--------------------------
Sys:ubuntu 16.04 LTS  
Hardware: CPU only (lab) ; GPU (note)

1.install
--------------------------
**Install virtualenv :**

* Install pip and Virtualenv.  
`sudo apt-get install python-pip python-dev python-virtualenv`  

* Create a Virtualenv environment.  
`virtualenv --system-site-packages ~/tensorflow`  

* Activate the Virtualenv environment and install TensorFlow in it.  
`source ~/tensorflow/bin/activate`  

**Install tensorflow :**
```bash
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl  

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl  
```

```bash
# Python 2
(tensorflow)$ pip install --upgrade $TF_BINARY_URL
# Python 3
(tensorflow)$ pip3 install --upgrade $TF_BINARY_URL
```

* After the install you will activate the Virtualenv environment each time you want to use TensorFlow.  

**Install cuda and cuDNN**
>https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#optional-install-cuda-gpus-on-linux

**Enable GPU Support**
```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```

2.MNIST For ML Beginners
-------------------------------
>https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html#the-mnist-data

i: the number of predict  
j: serial number of pixel
b(i): bias

x(j) : pixel of image ( tensor of [55000, 784] )   
y(n) : label of numbers ( tensor of [55000, 10] , one-hot vectors )  
//n:number of samples  

model(target): matrix W(i,j)  
predict : y = softmax( Wx + b )

### code (model) :  
```python
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
#tf.placeholder : 占位符
#None means that a dimension can be of any length (can import any numbers of image)

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#tf.zeros : initialize tensors full of zeros

y = tf.nn.softmax(tf.matmul(x,W) + b)
#tf.matmul(x,W) = W*x
```
### code (training)
```python
y_ = tf.placeholder(tf.float32, [None, 10])
#y_ : predict y

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#loss function : cross_entropy( 交叉熵 )
#tf.reduce_sum ： sums across all classes

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#gradient descent algorithm : 梯度下降算法
```

### code (run & evaluating)
```python
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#evaluating
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
