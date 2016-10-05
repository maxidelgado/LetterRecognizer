# coding=utf-8
import os

import cv2
import numpy as np
import tensorflow as tf

from Support import Prediction as pred

saved_model = os.path.join('SavedModels')


with tf.Graph().as_default():
    sess = tf.Session()

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 24])

        W = tf.Variable(tf.zeros([784, 24]))
        b = tf.Variable(tf.zeros([24]))

        y = tf.matmul(x, W) + b

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        # saver.restore(sess, saved_model)
        saver.restore(sess, saved_model + "/softmax_model")

        path_data_test = os.path.join('DataSet','data_test.npy')
        path_labels_test = os.path.join('DataSet','labels_test.npy')

        data_test = np.load(path_data_test)
        labels_test = np.load(path_labels_test)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Precisión: '+str(accuracy.eval(feed_dict={x: data_test, y_: labels_test})))

        for img in data_test:
            cl = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
            imgRes = img.reshape((28,28))
            cv2.imshow('letra',cv2.resize(imgRes,(100,100)))
            print pred.letra(cl[0])
            cv2.waitKey(0)


