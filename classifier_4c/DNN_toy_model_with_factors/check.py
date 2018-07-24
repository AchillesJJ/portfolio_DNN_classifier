# encoding: utf-8
import tensorflow as tf
import numpy as np
import scipy as sp
import cmath
import argparse
import math
import sys
import os

x = tf.placeholder(tf.float32, [None, 4], 'input-x')
y = tf.placeholder(tf.float32, [None, 4], 'input-y')
idx = tf.shape(tf.argmax(x))

for idx in range(idx[0]):
    pass

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data1 = [[1,4,2,6],[6,3,8,1],[8,9,0,2]]
    data2 = [[1,4,2,6],[6,3,8,1],[8,9,0,2]]
    # y = sess.run(tf.shape(idx), feed_dict={x: data1})
    # print(y)




































