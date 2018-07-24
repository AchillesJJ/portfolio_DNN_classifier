import tensorflow as tf
import numpy as np
import preprocessing
from preprocessing import SF_map, ism_data
from utility import *
import random
import pymysql
import re

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

# ism = ism_data('./ismdatabak_training', mode=0, batch_size=10)

# SF_map = SF_map()
# df = SF_map.SF_ls
# print(df)
# df.to_csv('SF_map.csv', header=True, index=None)
# df = pd.read_csv('SF_map.csv')
# print(df)

# database address
jydb_address = "117.122.223.35"
jydb_user_id = "zpy"
jydb_user_pwd = "Z1pe1y1@zpy"
jydb_db_name = "jydb02"
# on-line mode
conn = pymysql.connect(jydb_address, jydb_user_id, jydb_user_pwd, jydb_db_name)
# print(future_market_date('2018-04-09', 7, conn=conn))

ls = interval_filter('20180404', 7, conn)
cnt = 0
for file_name in ls:
    f = re.match(r'ismfactors\.(\d{4})-?(\d{2})-?(\d{2}).*', file_name)
    if f is None:
        continue
    else:
        with open(os.path.join('20180404', f.group())) as ism_file:
            for line in ism_file:
                cnt += 1

print(cnt-len(ls))









