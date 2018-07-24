# encoding: utf-8
import tensorflow as tf
import numpy as np
import scipy as sp
import cmath
import argparse
import math
import time
import sys
import os
import re
from sklearn.externals import joblib
# self-defined model
import preprocessing
from utility import *
from filter_gate import *

class ISMClassifier(object):
    """
    ISM classifier using cascading Deep-neural-network
    """
    
    def __init__(self, sess, N_logit, N_factor, learning_rate_base, num_train_set, batch_size):
        """
        doc is missing ... 
        """
        
        self.sess = sess
        self.N_logit = N_logit
        self.N_factor = N_factor
        self.learning_rate_base = learning_rate_base
        self.batch_size = batch_size
        self.num_train_set = num_train_set
        # exponential decay learning rate
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate_base,
            global_step=global_step,
            decay_steps=int(self.num_train_set/self.batch_size),
            decay_rate=0.99,
            staircase=True)
    
        # construct cascading DNN structure
        self.input_logit = tf.placeholder(tf.float32, [None, N_logit], 'input-logit')
        self.input_factor = tf.placeholder(tf.float32, [None, N_factor], 'input-factor')
        self.labels = tf.placeholder(tf.float32, [None, 4], 'labels')
        # (1) logit feature embedding layer
        l1 = tf.layers.dense(inputs=self.input_logit,
                             units=20,
                             activation=tf.nn.relu,
                             use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                             name='logit-embedding-layer')
        # (2) concatenate logits with factors
        self.input_concat = tf.concat([l1, self.input_factor], axis=1, name='input-concat')
        # (3) DNN classifier with concatenated inputs
        l2 = tf.layers.dense(inputs=self.input_concat,
                             units=200,
                             activation=tf.nn.relu,
                             use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                             name='DNN-hidden-layer-1')
        l3 = tf.layers.dense(inputs=l2,
                             units=40,
                             activation=tf.nn.relu,
                             use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                             name='DNN-hidden-layer-2')
        self.output = tf.layers.dense(inputs=l3,
                                      units=4,
                                      activation=tf.nn.relu,
                                      use_bias=True,
                                      kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                                      bias_initializer=tf.constant_initializer(0.0),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                                      name='output')
        
        # Optimization
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        self.loss = cross_entropy_mean + tf.losses.get_regularization_loss()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)                     

        # model accuracy
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # model saver
        self.saver = tf.train.Saver(max_to_keep=10)
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, input_logit, input_factor, labels):
        return self.sess.run([self.loss, self.accuracy, self.train_op],
                             feed_dict={self.input_logit: input_logit, 
                                        self.input_factor: input_factor, self.labels: labels})
    
    def validate(self, input_logit, input_factor, labels):
        return self.sess.run([self.output, self.accuracy],
                             feed_dict={self.input_logit: input_logit, 
                                        self.input_factor: input_factor, self.labels: labels})

    def evaluate(self, input_logit, input_factor):
        return self.sess.run(self.output, feed_dict={self.input_logit: input_logit,
                                                     self.input_factor: input_factor})


def main(args):
    """
    main function doc is missing ... 
    """
    with tf.Session() as sess:
        
        # training mode
        if args.mode == 0:
            ism_data = preprocessing.ism_data(ism_path=args.ism_training_path,
                                              interval=args.interval,
                                              batch_size=args.batch_size,
                                              scaler_ls=None,
                                              Rc=args.Rc,
                                              mode=0)
            # define cascade-DNN classifier
            N_logit = ism_data.N_logit
            N_factor = ism_data.N_factor
            num_train_set = ism_data.num_data
            classifier = ISMClassifier(sess, N_logit, N_factor, args.learning_rate_base, 
                                       num_train_set, args.batch_size)
            classifier.initialize()
            # checkpoint directory
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            else: # clear directory
                del_file(args.model_path)
            # iterate on epoch
            for ep in range(args.num_epoch):
                # iterate on batches
                for step in range(args.train_steps):
                    logit_batch, factor_batch, label_batch = ism_data.next_batch()
                    loss, accuracy, _ = classifier.train(logit_batch, factor_batch, label_batch)
                    if step % 1000 == 0:
                        print('epoch-{},step-{}: loss {}, accuracy {}'.format(ep, step, loss, accuracy))
                    if (step+1) % 10000 == 0:
                        classifier.saver.save(sess, os.path.join(args.model_path, 'model.ckpt'), global_step=step)
        # validation mode
        elif args.mode == 1:
            scaler_ls = joblib.load('./standard_scaler_list.pkl')
            ism_data = preprocessing.ism_data(ism_path=args.ism_validation_path,
                                              interval=args.interval,
                                              batch_size=args.batch_size,
                                              scaler_ls=scaler_ls,
                                              Rc=args.Rc,
                                              mode=1)
            # define cascade-DNN classifier
            N_logit = ism_data.N_logit
            N_factor = ism_data.N_factor
            num_train_set = ism_data.num_data
            classifier = ISMClassifier(sess, N_logit, N_factor, args.learning_rate_base, 
                                       num_train_set, args.batch_size)
            # restore model
            for file_name in os.listdir(args.model_path):
                f = re.match(r'model\.ckpt\-(\d*)\.data.*', file_name)
                if f is None:
                    continue
                else:
                    classifier.saver.restore(sess, os.path.join(args.model_path, 'model.ckpt-'+f.group(1)))
                    logit_all, factor_all, label_all = ism_data.validation_set()
                    output, accuracy = classifier.validate(logit_all, factor_all, label_all)
                    print('validation accuracy is {} after training steps {}'.format(accuracy, int(f.group(1))))
                    # output of gate operation
                    gate = filter_gate()
                    print(gate.stop_loss_gate(output, label_all))
        # testing mode
        elif args.mode == 2:
            pass
        
        # training while validation mode
        elif args.mode == 3:
            pass
            

# EXECUTE PROGRAM
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lrb', '--learning-rate-base', help='DNN learning rate base', type=float, default=1E-2)
    parser.add_argument('-bs', '--batch-size', help='training set batch size', type=int, default=100)
    parser.add_argument('--train-steps', help='number of training steps in each epoch', type=int, default=2000)
    parser.add_argument('--num-epoch', help='number of epoches', type=int, default=1)
    parser.add_argument('-mode', help='Training(0)/validation(1)/test(2)', type=int, choices=[0,1,2,3], default=0)
    parser.add_argument('--ism-training-path', help='path of ism training set directory', default='./ismdatabak_training')
    parser.add_argument('--ism-validation-path', help='path of ism validation set directory', default='./ismdatabak_validation')
    parser.add_argument('-interval', help='ism interval', type=int, default=7)
    parser.add_argument('--model-path', help='path for saving models', default='./model')
    parser.add_argument('-Rc', help='lower threshold of expected return', type=float, default=0.0)
    parser.add_argument('--eval-interval', help='num of interval steps between evaluations', type=int, default=2000)
    args= parser.parse_args()
    
    # run main program
    begin = time.clock()
    main(args)
    end = time.clock()
    print('total time cost is {}'.format(end-begin))

        



























