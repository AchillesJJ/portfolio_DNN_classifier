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
import ism_logger
import logging

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
        self.logger = logging.getLogger('ism_logger')
        
        # construct DNN structure
        self.inputs = tf.placeholder(tf.float32, [None, N_feature], 'inputs')
        self.labels = tf.placeholder(tf.float32, [None, 2], 'lables')
        self.ismcode = tf.placeholder(tf.string, [None, 1], 'ismcode') # isolate knot
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
                                 units=2,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3),
                                 name='output')
        
        # Optimization
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output)
        self.cross_entropy_mean = tf.reduce_mean(cross_entropy)
        self.regularization_loss = tf.losses.get_regularization_loss()
        self.loss = self.cross_entropy_mean + self.regularization_loss
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        
        # model accuracy
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # model saver
        self.saver = tf.train.Saver(max_to_keep=20)
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, inputs, labels):
        return self.sess.run([self.loss, self.accuracy, self.train_op], 
                             feed_dict={self.inputs: inputs, self.labels: labels})
    
    def accuracy_test(self, inputs, labels):
        return self.sess.run(self.accuracy,
                             feed_dict={self.inputs: inputs, self.labels: labels})
    
    def validate(self, inputs, labels):
        return self.sess.run([self.output, self.accuracy],
                             feed_dict={self.inputs: inputs, self.labels: labels})
    
    def evaluate(self, inputs):
        return self.sess.run([self.output, self.ismcode], 
                             feed_dict={self.inputs: inputs, self.ismcode: ismcode})
        

def main(args):
    """
    main function
    """
    # logger
    logger = logging.getLogger('ism_logger')
    
    with tf.Session() as sess:
            
        # training mode
        if args.mode == 0:
            logger.info('factor based DNN classifier, training mode begins ...')
            ism_data = preprocessing.ism_data(ism_path=args.ism_training_path,
                                         interval=args.interval,
                                         batch_size=args.batch_size,
                                         mode=0)
            N_feature = ism_data.N_feature
            num_train_set = ism_data.num_data
            # define DNN classifier
            classifier = ISMClassifier(sess, N_feature, args.learning_rate_base, 
                                       ism_data.num_data, args.batch_size, args.Rc)
            classifier.initialize()
            # checkpoint directory
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            else:
                del_file(args.model_path)
                
            # iterate on epoch
            num_steps = int(num_train_set/args.batch_size)
            index = []
            for ep in range(args.num_epoch):
                # iterate on batches
                for step in range(num_steps):
                    input_batch, label_batch = ism_data.next_batch()
                    loss, accuracy, _ = classifier.train(input_batch, label_batch)
                print('epoch-{}: loss {}, accuracy {}'.format(ep, loss, accuracy))
                logger.info('epoch-{0:n}: loss {1:.4f}, accuracy {2:.3f}'.format(ep, loss, accuracy))    
            # save model
                if ep > 10:
                    input_test, label_test = ism_data.next_batch(batch_size=2000)
                    accuracy_test = classifier.accuracy_test(input_test, label_test)
                    classifier.saver.save(sess, os.path.join(args.model_path, 'model.ckpt'), global_step=ep)
                    index.append([ep, loss, accuracy_test])
            index = pd.DataFrame(index, columns=['epoch', 'loss', 'accuracy'])
            index.to_csv(os.path.join(args.model_path, 'index.csv'), index=None)        
                    
        # validation mode
        elif args.mode == 1:
            logger.info('factor based DNN classifier, validation mode begins ...')
            scaler_ls = joblib.load('./standard_scaler_list.pkl')
            ism_data = preprocessing.ism_data(ism_path=args.ism_validation_path,
                                         interval=args.interval,
                                         batch_size=args.batch_size,
                                         scaler_ls=scaler_ls,
                                         mode=1)
            # define DNN classifier
            N_feature = ism_data.N_feature
            num_train_set = ism_data.num_data
            classifier = ISMClassifier(sess, N_feature, args.learning_rate_base, 
                                       ism_data.num_data, args.batch_size, args.Rc)
            
            for file_name in os.listdir(args.model_path):
                f = re.match(r'model\.ckpt\-(\d*)\.data.*', file_name)
                if f is None:
                    continue
                else:
                    classifier.saver.restore(sess, os.path.join(args.model_path, 'model.ckpt-'+f.group(1)))
                    input_all, label_all = ism_data.validation_set()
                    output, accuracy = classifier.validate(input_all, label_all)
                    print('validation accuracy is {} after training steps {}'.format(accuracy, int(f.group(1))))
                    # output of gate operation
                    gate = filter_gate()
                    x1, x2, x3, x4 = gate.stop_loss_gate(output, label_all)
                    print(gate.stop_loss_gate(output, label_all))
                    logger.info('validation accuracy is {0:.3f} for model-{1:n}, original win-ratio = {2:.3f} and num = {3:n}, and filtering win-ratio = {4:.3f} and num = {5:n}' \
                                .format(accuracy, int(f.group(1)), x1, x2, x3, x4))
                    
        # testing mode
        else:
            logger.info('dense encoding DNN classifier, evaluation mode begins ...')
            ism_data = preprocessing.ism_data(ism_path=args.ism_evaluation_path,
                                              interval=args.interval,
                                              batch_size=args.batch_size,
                                              scaler_ls=scaler_ls,
                                              mode=2)
            # define DNN classifier
            N_feature = ism_data.N_feature
            num_train_set = ism_data.num_data
            classifier = ISMClassifier(sess, N_feature, args.learning_rate_base, num_train_set, args.batch_size)
            
            # restore model
            index = pd.read_csv(os.path.join(args.model_path, 'index.csv'))
            residue = []
            # check output directory
            if not os.path.exists(args.ismcode_output_path):
                os.mkdir(args.ismcode_output_path)
            else:
                del_file(args.ismcode_output_path)
                
            for file_name in os.listdir(args.model_path):
                f = re.match(r'model\.ckpt\-(\d*)\.data.*', file_name)
                if f is None:
                    continue
                else:
                    classifier.saver.restore(sess, os.path.join(args.model_path, 'model.ckpt-'+f.group(1)))
                    logit_all, ismcode_all = ism_data.evaluation_set()
                    output, ismcode = classifier.evaluate(logit_all, ismcode_all)
                    # output of gate operation
                    gate = filter_gate()
                    code_ls = gate.filter_output(output, ismcode)
                    residue.append([int(f.group(1)), float(len(code_ls)/len(output))])
            
            # choose the best model to generate
            residue = pd.DataFrame(residue, columns=['epoch', 'residue_ratio'])
            index_all = pd.merge(index, residue, on='epoch')
            index_all = index_all.sort_values(by='epoch').reset_index()
            index_all.to_csv(os.path.join(args.model_path, 'index_all.csv'), index=None)
            index_ranking = np.zeros((len(index_all), 4), dtype=np.float64)
            index_ranking[:, 0] = index_all['loss'].argsort().values
            index_ranking[:, 1] = index_all['accuracy'].argsort().values
            index_ranking[:, 2] = index_all['residue_ratio'].argsort().values
            index_ranking[:, 3] = 0.4*index_ranking[:,0] + 0.4*index_ranking[:,1] + 0.2*index_ranking[:,2]
            mode_id = index_all['epoch'].iloc[np.argmin(index_ranking[:, 3])]
            classifier.saver.restore(sess, os.path.join(args.model_path, 'model.ckpt-'+str(mode_id)))
            logit_all, ismcode_all = ism_data.evaluation_set()
            output, ismcode = classifier.evaluate(logit_all, ismcode_all)
            # output of gate operation
            gate = filter_gate()
            code_ls = gate.filter_output(output, ismcode)
            np.savetxt(os.path.join(args.ismcode_output_path, 'filter_ismcode.txt'), code_ls, fmt='%s') 
            print('the chosen best model is model-{}'.format(mode_id))
            logger.info('the chosen best model is model-{}'.format(mode_id))   
        

# EXECUTE PROGRAM    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lrb', '--learning-rate-base', help='DNN learning rate base', type=float, default=1E-2)
    parser.add_argument('-bs', '--batch-size', help='training set batch size', type=int, default=100)
    parser.add_argument('--num-epoch', help='number of epoches', type=int, default=10)
    parser.add_argument('-mode', help='Training(0)/validation(1)/test(2)', type=int, choices=[0,1,2,3], default=0)
    parser.add_argument('--ism-training-path', help='path of ism training set directory', default='./ismdatabak_training')
    parser.add_argument('--ism-validation-path', help='path of ism validation set directory', default='./ismdatabak_validation')
    parser.add_argument('--ism-evaluation-path', help='path of ism evaluation set directory', default='./ismdatabak_evaluation')
    parser.add_argument('--ismcode-output-path', help='path of output ismcode after filtering', default='./filter_output')
    parser.add_argument('-interval', help='ism interval', type=int, default=7)
    parser.add_argument('--model-path', help='path for saving models', default='./model')
    parser.add_argument('-Rc', help='lower threshold of expected return', type=float, default=0.0)
    parser.add_argument('--eval-interval', help='num of interval steps between evaluations', type=int, default=2000)
    args = parser.parse_args()
    
    # run main program
    main(args)






















































