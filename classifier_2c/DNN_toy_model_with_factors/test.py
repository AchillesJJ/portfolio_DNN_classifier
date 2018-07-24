import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing
from preprocessing import ism_list, ism_factor, ism_data
from sklearn import preprocessing
from sklearn.externals import joblib
from utility import *
import random
import requests
import time
import re
import multiprocessing
import matplotlib.pyplot as plt

# def structure(ism_input):
#     """
#     read in ism_input ASCII text file and transforms to structured data
#     ism_input: string, file path of ism file
#     return: a structured data set (numpy list)
#     """
#     f = open(ism_input, 'r')
#     data = []
#     for line in f:
#         line = line.replace('\"', '')
#         line = line.replace('],[', '];[')
#         line = line.strip()
#         line = line.replace(']', '')
#         line = line.replace('[', '')
#         line = line.split(';')
#         line[0] = line[0].split('|')
#         ls = list(map(lambda x: x.split(','), line[1:]))
#         ls = list(map(lambda x: list(map(lambda y: y.split('|'), x)), ls))
#         line[1:] = ls
#         data.append(line)
#     data = np.array(data[1:])    
# 
#     return data
# 
# data = structure('ismlist.20180227')
# # for ism_pack in data:
# #     print(ism_pack)
# 
# preAlloc = preprocessing()
# df = preAlloc.SF_ls
# 
# ism_pack = data[0][1:]
# ls = map(lambda x: [y[1] for y in x[1:]], ism_pack)
# ls = flatten(ls)
# print(ls)
# sparse_ls = []
# for i in ls:
#     sparse_ls.append(df[df['SecuCode']==i].index.tolist()[0])
# 
# print(sparse_ls)

# preAlloc = preprocessing()
# df = preAlloc.SF_ls
# 
# ism_data = ism_data('./ismdatabak', df)


# with open('./sparse_code.txt', 'w') as f:
#     f.write(str(data))
# f.close()

# sparse_code = [[1, 3, 5, 7],[2, 4, 6, 8]]
# dense_code = tf.sparse_to_dense(sparse_indices=sparse_code,
#                                 output_shape=[10],
#                                 sparse_values=1.0)
# 
# with tf.Session() as sess:
#     print(sess.run(dense_code))

# preAlloc = preprocessing()
# df = preAlloc.SF_ls
# ism_data = ism_data('./', df, N_feature=preAlloc.N_feature, batch_size=1)
# a, b = ism_data.dense_encoding(ism_data.sparse_code)
# inputs, label = ism_data.next_batch()
# print(inputs, label)

# x = tf.placeholder(tf.float32, [None, 3521])
# y = tf.placeholder(tf.float32, [None, 3521])
# z = x + y
# 
# with tf.Session() as sess:
#     preAlloc = preprocessing()
#     df = preAlloc.SF_ls
#     ism_data = ism_data('./', df, preAlloc.N_feature, batch_size=1)
#     inputs, label = ism_data.next_batch()
# 
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(z, feed_dict={x: inputs, y: inputs}))

# pre= preprocessing()
# ism_data = ism_data(ism_path='./ismdatabak', SF_ls=pre.SF_ls, Rc=0.0, N_feature=pre.N_feature,batch_size=10)
# df = pre.SF_ls
# df.to_csv('./SF_map.csv', header=None, index=None)

# with open('./status.txt', 'r') as f:
#     cnt = 0
#     for line in f:
#         line = line.split('],[')
#     print(line)

# f = open('./ismdatabak_training/ismfactors.20180313', 'r')
# cnt = 0
# for line in f:
#     line = line.replace('\"', '')
#     line = line.replace('[', '')
#     line = line.replace(']', '')
#     line = line.strip()
#     line = line.split('|')
#     if cnt > 0:
#         data = pd.DataFrame([[line[0]] + line[5:]], columns=col_name)
#         df = df.append(data, ignore_index=True)
#     else:
#         col_name = [line[0]] + line[5:]
#         df = pd.DataFrame(columns=col_name)
#     cnt += 1
# 
# for col in col_name[1:]:
#     df[col] = df[col].astype(float)

# factor = ism_factor('./ismdatabak_training')
# df = factor.ism_factor
# df.pop('ismcode')
# # df_scaled = preprocessing.scale(df)
# x = np.reshape(df['hurst'].values, (-1, 1))
# df['hurst'] = np.reshape(preprocessing.scale(x), (-1))
# print(df['msf'])

# label = ism_list('./ismdatabak_training', Rc=0.0)
# factor = ism_factor('./ismdatabak_training')
# df1 = label.labels()
# # df2 = factor.structure('./ismdatabak_training/ismfactors.20180313')
# df3, _ = factor.standard_factor()
# 
# df = pd.merge(df1, df3, on='ismcode')
# print(df['ismcode'])

# ism = ism_data('./ismdatabak_training', 100)
# inputs, labels = ism.next_batch()
# print(labels[0:5])
# print(inputs[0:5])

# ls = ism_factor('./ismdatabak_training')
# df, _ = ls.standard_factor()
# print(df[['ismcode', 'hurst']])
# df[['ismcode', 'hurst']].to_csv('./ism_factor.csv', header=None, index=None)

# ls = ism_data('./ismdatabak_training', 100)
# data = ls.data
# data.to_csv('./ism_data.csv', header=True, index=None)

# df = pd.read_csv('./ism_data.csv')
# x = df['ie'].tolist()
# y = df['hurst'].tolist()
# label = df[['C1', 'C2']].as_matrix()
# color_ls = []
# for i in label:
#     if i[0] == 1 or i[1] == 1:
#         color_ls.append('r')
#     else:
#         color_ls.append('b')
# 
# plt.figure()
# plt.scatter(x, y, marker='o', c=color_ls, s=0.2)
# plt.show()

# label = ism_list('./ismdatabak_training', Rc=0.0)
# df1 = label.labels()
# print(df1)
# df1.to_csv('./ism_label.csv', header=None, index=None)

# ls = []
# cnt = 0
# for file_name in os.listdir('./ismdatabak_training'):
#     cnt += 1
#     f = re.match(r'ismfactors\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
#     if f is None:
#         continue
#     else:
#         f2 = os.path.join('ismdatabak_training',
#                           re.sub('ismfactors', 'ismlist', f.group()))
#         if not os.path.isfile(f2):
#             continue
#         else:
#             ls.append(f2)
# 
# print(len(ls), cnt)

scaler_ls = joblib.load('standard_scaler_list.pkl')
ism = ism_data('./ismdatabak_validation', batch_size=10, scaler_ls=scaler_ls, mode=1)
print(ism.data)
    
        
























