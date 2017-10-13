# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer




'''
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

############################# end ###

#sess = tf.Session()
#sess.run(init)

#for step in range(2000):
#    sess.run(train)
#    if step%100 == 0:
#        print(step,sess.run(Weights),sess.run(biases))


matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

state = tf.Variable(0,name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
'''
'''
# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
'''
mnist = input_data.read_data_sets('mnist_data',one_hot=True)

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):


    Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
    biaies = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
    Wx_plus_b = tf.matmul(inputs,Weights)+biaies
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    tf.histogram_summary(layer_name + '/outputs', outputs)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global predition
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32,[None, 784])
ys = tf.placeholder(tf.float32,[None, 10])
x_image = tf.reshape(xs, [-1, 28, 28, 1])   ############## 负一负一负一 ###############

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)   # 7*7*64

W_f1 = weight_variable([7*7*64, 1024])
b_f1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_f1) + b_f1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_f2 = weight_variable([1024, 10])
b_f2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_f2) + b_f2)




cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.scalar_summary('loss',cross_entropy)
train_step =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i%50==0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))





