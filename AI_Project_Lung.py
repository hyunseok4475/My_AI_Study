import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os

# print(os.listdir("../input"))

# mainDIR = os.listdir('../input/chest_xray/chest_xray')
# print(mainDIR)

# train_folder='../input/chest_xray/chest_xray/train/'
# val_folder = '../input/'
# test_folder = ''

# os.listdir(train_folder)
# train_n = train_folder+'NORMAL/'
# train_p = train_folder+'PNEUMONIA/'
# train_c = train_folder+'CANCER/'

# print(len(os.listdir(train_n)))
# rand_norm= np.random.randint(0,len(os.listdir(train_n)))
# norm_pic = os.listdir(train_n)[rand_norm]
# print('normal picture title: ',norm_pic)

# norm_pic_address = train_n+norm_pic

# rand_p = np.random.randint(0,len(os.listdir(train_p)))

# sic_pic =  os.listdir(train_p)[rand_norm]
# sic_address = train_p+sic_pic
# print('pneumonia picture title:', sic_pic)

# norm_load = Image.open(norm_pic_address)
# sic_load = Image.open(sic_address)

# f = plt.figure(figsize= (10,6))
# a1 = f.add_subplot(1,2,1)
# img_plot = plt.imshow(norm_load)
# a1.set_title('Normal')

# a2 = f.add_subplot(1, 2, 2)
# img_plot = plt.imshow(sic_load)
# a2.set_title('Pneumonia')

# epoch = 30
# wide = 500
# height = 500

# #정상 폐 이미지 해상도 통일 및 파일 이름 변경
# Y = tf.placeholder(tf.float32, [None, 3])

# L1 = tf.layers.conv2d(X, 2048, [3, 3])
# L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
# L1 = tf.layers.dropout(L1, 0.7, is_training)

# L2 = tf.layers.conv2d(L1, 1024, [3, 3])
# L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
# L2 = tf.layers.dropout(L2, 0.7, is_training)

# L3 = tf.contrib.layers.flatten(L2)
# L3 = tf.layers.dense(L3, 256, activation = tf.nn.relu)
# L3 = tf.layers.dropout(L3, 0.5, is_training)

# model = tf.layers.dense(L3, 10, activation = None)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels=Y))
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.8).minimize(cost)

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# batch_size = 100

# for i in range(epoch):
#     total_cost = 0
    