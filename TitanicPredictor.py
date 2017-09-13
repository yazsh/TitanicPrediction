import pandas as pd
import numpy as np
import tensorflow as tf
from HelperFunctions import *
from TitanicDataCleaning import *
input_features = 2433
hidden1 = 10
hidden2 = 10
hidden3 = 10
hidden4 = 1

learning_rate=.001

weights = dict(
            w1=tf.Variable(tf.random_normal([input_features, hidden1]),"adfasf"),
            w2=tf.Variable(tf.random_normal([hidden1, hidden2]), "222222"),
            w3=tf.Variable(tf.random_normal([hidden2, hidden3])),
            w4=tf.Variable(tf.random_normal([hidden3, hidden4]))
            )

biases = dict(
            b1=tf.Variable(tf.zeros([hidden1])),
            b2=tf.Variable(tf.zeros([hidden2])),
            b3=tf.Variable(tf.zeros([hidden3])),
            b4=tf.Variable(tf.zeros([hidden4]))
            )

x = tf.placeholder("float32", [None,input_features])
layer = create_layer(x, weights['w1'], biases['b1'])
layer = create_layer(layer, weights['w2'], biases['b2'])
layer = create_layer(layer, weights['w3'], biases['b3'])
Z4 = create_layer(layer, weights['w4'], biases['b4'], None)

y = tf.placeholder(dtype="float32",shape=[None,1])

# cost = tf.reduce_mean(tf.cast(tf.equal(Z4, y), "float32"))
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z4, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(Z4, feed_dict={x: train_feature, y: train_labels})[:10])
    for i in range(0, 10):

        _, c = sess.run([optimizer, cost], feed_dict={x: train_feature, y: train_labels})
        # print(sess.run(Z4, feed_dict={x: train_feature, y: train_labels})[:10])
        print("Iteration " + str(i) + ":  " + str(c))

