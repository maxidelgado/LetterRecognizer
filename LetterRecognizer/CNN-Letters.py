import tensorflow as tf
import numpy as np
import os
from Support import NpyUtils as nputil

save_model = os.path.join('SavedModels')

path_data_train = os.path.join('DataSet','data_train.npy')
path_labels_train = os.path.join('DataSet','labels_train.npy')
path_data_test = os.path.join('DataSet','data_test.npy')
path_labels_test = os.path.join('DataSet','labels_test.npy')

batch_xs = np.load(path_data_train)
batch_ys = np.load(path_labels_train)

data_test = np.load(path_data_test)
labels_test = np.load(path_labels_test)

#Inicia una sesion de Tensorflow que llama a un backend en C++ muy eficiente.
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 24])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 24])
b_fc2 = bias_variable([24])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

dataSet = nputil.DataSet()
saver = tf.train.Saver()
for i in range(20000):
    mini_batch_xs, mini_batch_ys = dataSet.next_batch(batch_xs, batch_ys, 50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                x:mini_batch_xs, y_: mini_batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        if i%5000 == 0:
            saver.save(sess, save_path=save_model+'/CNN-model-',global_step=i)
    train_step.run(feed_dict={x: mini_batch_xs, y_: mini_batch_ys, keep_prob: 0.5})

saver.save(sess, save_path=save_model+'/CNN-model-',global_step=20000)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: data_test, y_: labels_test, keep_prob: 1.0}))