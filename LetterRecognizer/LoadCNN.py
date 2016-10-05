# coding=utf-8
import os

import cv2
import numpy as np
import tensorflow as tf

import Support.NpyUtils as nputil
from Support import Prediction as pred

saved_model = os.path.join('SavedModels')

with tf.Graph().as_default():
    sess = tf.Session()

    with tf.Session() as sess:
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

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 24])
        b_fc2 = bias_variable([24])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        # saver.restore(sess, saved_model)
        saver.restore(sess, saved_model + "/CNN-model--10000")

        path_data_test = os.path.join('DataSet','data_test.npy')
        path_labels_test = os.path.join('DataSet','labels_test.npy')

        data_test = np.load(path_data_test)
        labels_test = np.load(path_labels_test)

        data_test, labels_test = nputil.random_shuffle_twoArrays(data_test, labels_test)

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: data_test, y_: labels_test, keep_prob: 1.0}))

        for img in data_test:
            cl = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [img], keep_prob: 1.0})
            imgRes = img.reshape((28,28))
            cv2.imshow('letra',cv2.resize(imgRes,(100,100)))
            print pred.letra(cl[0])
            cv2.waitKey(0)


