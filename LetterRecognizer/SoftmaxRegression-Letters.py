# coding=utf-8
import tensorflow as tf
import numpy as np
import os
from Support import NpyUtils as nputil
import cv2

path_data_train = os.path.join('DataSet','data_train.npy')
path_labels_train = os.path.join('DataSet','labels_train.npy')
path_data_test = os.path.join('DataSet','data_test.npy')
path_labels_test = os.path.join('DataSet','labels_test.npy')

data_train = np.load(path_data_train)
labels_train = np.load(path_labels_train)

data_test = np.load(path_data_test)
labels_test = np.load(path_labels_test)

saved_model = os.path.join('SavedModels')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 24])

W = tf.Variable(tf.zeros([784,24]))
b = tf.Variable(tf.zeros([24]))

sess.run(tf.initialize_all_variables())

y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

dataSet = nputil.DataSet()
saver = tf.train.Saver()

for i in range(1000):
    mini_batch_xs, mini_batch_ys = dataSet.next_batch(data_train, labels_train, 100)
    train_step.run(feed_dict={x: data_train, y_: labels_train})
    if i % 200 == 0:
        print("step: %d" % i)


saver.save(sess, saved_model+'/softmax_model')
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Precisión: '+str(accuracy.eval(feed_dict={x: data_test, y_: labels_test})))
