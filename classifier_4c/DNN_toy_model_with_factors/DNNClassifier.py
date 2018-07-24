# encoding: utf-8
import tensorflow as tf
import numpy as np
import scipy as sp
import cmath
import argparse
import math
import sys
import os
import re
from sklearn.externals import joblib
# self-defined modules
import preprocessing
from utility import *
from filter_gate import *

class ISMClassifier(object):
    """
    ISM classifier using Deep-neural-network
    """
    
    def __init__(self, sess, N_feature, learning_rate_base, num_train_set, batch_size, rc=None):
        
        self.sess = sess
        self.N_feature = N_feature
        self.batch_size = batch_size
        self.rc = rc
        self.learning_rate_base = learning_rate_base
        # exponential decay learning rate
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate_base,
            global_step=global_step,
            decay_steps=int(num_train_set/batch_size),
            decay_rate=0.99,
            staircase=True)
        
        # construct DNN structure
        self.inputs = tf.placeholder(tf.float32, [None, N_feature], 'inputs')
        self.labels = tf.placeholder(tf.float32, [None, 4], 'lables')
        l1 = tf.layers.dense(inputs=self.inputs,
                             units=200,
                             activation=tf.nn.relu,
                             use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                             name='layer-1')
        l2 = tf.layers.dense(inputs=l1,
                             units=60,
                             activation=tf.nn.relu,
                             use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                             name='layer-2')
        l3 = tf.layers.dense(inputs=l2,
                             units=20,
                             activation=tf.nn.relu,
                             use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                             bias_initializer=tf.constant_initializer(0.0),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                             name='layer-3')
        self.output = tf.layers.dense(inputs=l3,
                                 units=4,
                                 activation=None,
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
        self.saver = tf.train.Saver(max_to_keep=5)
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, inputs, labels):
        return self.sess.run([self.loss, self.accuracy, self.train_op], 
                             feed_dict={self.inputs: inputs, self.labels: labels})
    
    def validate(self, inputs, labels):
        return self.sess.run([self.output, self.accuracy],
                             feed_dict={self.inputs: inputs, self.labels: labels})
    
    def evaluate(self, inputs):
        return self.sess.run(self.output, feed_dict={self.inputs: inputs})
        
    def get_learning_rate(self):
        return self.sess.run(self.learning_rate)
        

def main(args):
    """
    main function
    """
    
    with tf.Session() as sess:
            
        # training mode
        if args.mode == 0:
            ism = preprocessing.ism_data(ism_path=args.ism_training_path,
                                         interval=args.interval,
                                         batch_size=args.batch_size,
                                         mode=0)
            N_feature = ism.N_feature
            # define DNN classifier
            classifier = ISMClassifier(sess, N_feature, args.learning_rate_base, 
                                       ism.num_data, args.batch_size, args.Rc)
            classifier.initialize()
            # checkpoint directory
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            else:
                del_file(args.model_path)
            # iterate on epoch
            for ep in range(args.num_epoch):
                # iterate on batches
                for step in range(args.train_steps):
                    input_batch, label_batch = ism.next_batch()
                    loss, accuracy, _ = classifier.train(input_batch, label_batch)
                    output, _ = classifier.validate(input_batch, label_batch)
                    learning_rate = classifier.get_learning_rate()
                    if step < 999 and (step+1) % 50 == 0:
                        print('epoch-{},step-{}: loss {}, accuracy {} and learning rate {}'\
                              .format(ep, step, loss, accuracy, learning_rate))
                    elif step >= 999 and (step+1) % 1000 == 0:
                        print('epoch-{},step-{}: loss {}, accuracy {} and learning rate {}'\
                              .format(ep, step, loss, accuracy, learning_rate))
                    if (step+1) % 10000 == 0:
                        classifier.saver.save(sess, os.path.join(args.model_path, 'model.ckpt'), global_step=step)
        # validation mode
        elif args.mode == 1:
            scaler_ls = joblib.load('./standard_scaler_list.pkl')
            ism = preprocessing.ism_data(ism_path=args.ism_validation_path,
                                         interval=args.interval,
                                         batch_size=args.batch_size,
                                         scaler_ls=scaler_ls,
                                         mode=1)
            N_feature = ism.N_feature
            # define DNN classifier
            classifier = ISMClassifier(sess, N_feature, args.learning_rate_base, 
                                       ism.num_data, args.batch_size, args.Rc)
            for file_name in os.listdir(args.model_path):
                f = re.match(r'model\.ckpt\-(\d*)\.data.*', file_name)
                if f is None:
                    continue
                else:
                    classifier.saver.restore(sess, os.path.join(args.model_path, 'model.ckpt-'+f.group(1)))
                    input_all, label_all = ism.validation_set()
                    output, accuracy = classifier.validate(input_all, label_all)
                    print('validation accuracy is {} after training steps {}'.format(accuracy, int(f.group(1))))
                    # output after gate operation
                    gate = filter_gate()
                    print(gate.stop_loss_gate(output, label_all))
        # testing mode
        else:
            pass
        

# EXECUTE PROGRAM    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lrb', '--learning-rate-base', help='DNN learning rate base', type=float, default=1E-2)
    parser.add_argument('-bs', '--batch-size', help='training set batch size', type=int, default=100)
    parser.add_argument('--train-steps', help='number of training steps in each epoch', type=int, default=2000)
    parser.add_argument('--num-epoch', help='number of epoches', type=int, default=1)
    parser.add_argument('-mode', help='Training(0)/validation(1)/test(2)', type=int, choices=[0,1,2], default=0)
    parser.add_argument('--ism-training-path', help='path of ism training data directory', default='./ismdatabak_training')
    parser.add_argument('--ism-validation-path', help='path of ism validation data directory', default='./ismdatabak_validation')
    parser.add_argument('-interval', help='ism interval', type=int, default=7)
    parser.add_argument('--model-path', help='path for saving models', default='./model')
    parser.add_argument('-Rc', help='lower threshold of expected return', type=float, default=0.0)
    args = parser.parse_args()
    
    # run main program
    main(args)






















































