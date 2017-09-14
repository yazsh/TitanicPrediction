import pandas as pd
import numpy as np
import tensorflow as tf
from HelperFunctions import *
from TitanicDataCleaning import *


input_features = 2433
hidden1 = 20
hidden2 = 40
hidden3 = 20
hidden4 = 1

learning_rate =.01
beta = .001
weights = dict(
            w1=tf.Variable(tf.random_normal([input_features, hidden1]),),
            w2=tf.Variable(tf.random_normal([hidden1, hidden2]),),
            w3=tf.Variable(tf.random_normal([hidden2, hidden3])),
            w4=tf.Variable(tf.random_normal([hidden3, hidden4]))
            )

biases = dict(
            b1=tf.Variable(tf.zeros([hidden1])),
            b2=tf.Variable(tf.zeros([hidden2])),
            b3=tf.Variable(tf.zeros([hidden3])),
            b4=tf.Variable(tf.zeros([hidden4]))
            )

global_step = tf.Variable(0, trainable=False)

decay_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.96, staircase=True)


x = tf.placeholder("float32", [None,input_features])

layer = create_layer(x, weights['w1'], biases['b1'], tf.tanh)
layer = create_layer(layer, weights['w2'], biases['b2'], tf.tanh)
layer = tf.nn.dropout(layer, keep_prob=.9)
layer = create_layer(layer, weights['w3'], biases['b3'], tf.tanh)
Z4 = create_layer(layer, weights['w4'], biases['b4'])

y = tf.placeholder(dtype="float32",shape=[None, 1])

regularizer = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3']) + tf.nn.l2_loss(weights['w4'])

# cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=Z4))
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z4, labels=y) + (beta * regularizer))
optimizer = tf.train.AdamOptimizer(decay_learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0, 400):

        _, c = sess.run([optimizer, cost], feed_dict={x: train_feature, y: train_labels})
        print("Iteration " + str(i) + ":  " + str(c))

    prediction = tf.cast(tf.round(tf.sigmoid(Z4)),"int32")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,train_labels), "float32"))
    accuracy2 = tf.reduce_mean(tf.cast(tf.equal(prediction,dev_labels), "float32"))

    print(sess.run(accuracy, feed_dict={x:train_feature, y:train_labels}))
    print(sess.run(accuracy2, feed_dict={x:dev_feature, y:dev_labels}))
    test = sess.run(prediction, feed_dict={x:test_feature})
    frame = pd.DataFrame(test)
    frame.index += 892
    frame.index.name = "PassengerId"
    frame.to_csv("/Users/yazen/Desktop/datasets/Titanic/prediction.csv", header=["Survived"])


