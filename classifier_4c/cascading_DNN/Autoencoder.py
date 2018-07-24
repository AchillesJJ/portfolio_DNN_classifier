# encoding: utf-8
import tensorflow as tf
import numpy as np
import scipy as sp
import cmath
import argparse
import math
import sys
import os


class Autoencoder(object):
    """
    auto-encode one-hot feature to low-dimension representation
    """
    
    def __init__(self, sess, N_feature, learning_rate, batch_size):
        self.sess = sess
        self.N_feature = N_feature
        self.lr = learning_rate
        self.batch_size = batch_size
        
        # construct symmetric autoencoder NN
        self.inputs = tf.placeholder(tf.float32, [None, N_feature], 'inputs')
        self.labels = tf.placeholder(tf.float32, [None, N_feature], 'labels')
        l1 = tf.layers.dense(inputs=self.inputs,
                             units=300,
                             activation=tf.sigmoid,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             name='layer-1')
        self.l2 = tf.layer.dense(inputs=l1,
                                 units=50,
                                 activation=tf.sigmoid,
                                 kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 name='layer-2')
        l3 = tf.layer.dense(inputs=self.l2,
                                 units=300,
                                 activation=tf.sigmoid,
                                 kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 name='layer-3')
        self.output = tf.layers.dense(inputs=l3,
                                 units=self.N_feature,
                                 activation=tf.sigmoid,
                                 kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 name='layer-3')
        
        # Optimization
        global_step = tf.Variable(0, trainable=False)
        self.loss = tf.reduce_mean(tf.square(self.output, self.labels))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=global_step)
        
        # model saver
        self.saver = tf.train.Saver(max_to_keep=5)
        
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, inputs, labels):
        return self.sess.run([self.loss, self.train_op],
                             feed_dict={self.inputs: inputs, self.labels: labels})    
        
    def encode(self, inputs):
        return self.sess.run(self.l2, feed_dict={self.inputs: inputs})
    
    def decode(self, inputs):
        return self.sess.run(self.output, feed_dict={self.inputs: inputs})

def main(args):
    """
    main function
    """
    pass
    
    
    
if __name__ == '__main__':
    pass


































