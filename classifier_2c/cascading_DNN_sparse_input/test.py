import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing
from preprocessing import SF_map, ism_data
import sklearn
from utility import *
import random
import re

# ismlist = preprocessing.ism_list('./ismdatabak_training')
# # logit_labels = ismlist.ism_logit_and_labels('./ismdatabak_training/ismlist.20180313.1')
# logit_labels = ismlist.logit_and_labels()
# ismfactor = preprocessing.ism_factor('./ismdatabak_training')
# std_factor = ismfactor.standard_factor()
# ismdata = preprocessing.ism_data('./ismdatabak_training', 10)
# logit_batch, factor_batch, label_batch = ismdata.next_batch()
# print(factor_batch)
# print(label_batch)

# X = tf.Variable(tf.truncated_normal([10,5], stddev=1.0, mean=0.0))
# sp_indices = tf.placeholder(dtype=tf.int64)
# sp_shape = tf.placeholder(dtype=tf.int64)
# sp_ids_val = tf.placeholder(dtype=tf.int64)
# sp_weight_val = tf.placeholder(dtype=tf.float32)
# 
# sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
# sp_weights = tf.SparseTensor(sp_indices, sp_weight_val, sp_shape)
# 
# y = tf.nn.embedding_lookup_sparse(X, sp_ids, sp_weights, combiner="sum")
# z = tf.nn.embedding_lookup_sparse(X, sp_ids, None, combiner="sum")
# 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     res_y = sess.run(y, feed_dict={sp_indices: [[0,0],[0,1],[1,0],[1,2]],
#                                  sp_shape: [1,3],
#                                  sp_ids_val: [0,1,0,2],
#                                  sp_weight_val: [1.0,2.0,1.0,1.0]})
#     res_z = sess.run(z, feed_dict={sp_indices: [[0,0],[0,1],[1,0],[1,2]],
#                                  sp_shape: [1,3],
#                                  sp_ids_val: [0,1,0,2]})
#     print(sess.run(X))
#     print('')
#     print(res_y)
#     print('')
#     print(res_z)
# 
# # X = tf.sparse_placeholder(tf.float32, shape=[None, 5])
# # w = tf.get_variable('w', [5, 3], dtype=tf.float32)
# # z = tf.matmul(X, w)
# # y = tf.sparse_reduce_sum(X, 1)
# # 
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     indices = [[0,0],[0,1],[1,2],[1,3]]
# #     values = [1.0,2.0,3.0,4.0]
# #     shape = [2,5]
# #     res = sess.run(z, feed_dict={X: (indices, values, shape)})
# #     print(res)

# ismpath = './ismdatabak_training'
# batch_size = 5
# ismdata = preprocessing.ism_data(ismpath, batch_size)
# indices, idsval, shape, factor, labels = ismdata.next_batch()
# print(indices)
# print(idsval)
# print(shape)
# print(factor)
# print(labels)

# ismlist = preprocessing.ism_list('./ismdatabak_training')
# output = ismlist.logit_and_labels()
# for line in output:
#     print(line)

factor = preprocessing.ism_factor('./ismdatabak_training')
ls, _ = factor.standard_factor()
data = np.reshape(ls['sd'].values, (-1, 1))
data = sklearn.preprocessing.scale(data)
print(data)























































