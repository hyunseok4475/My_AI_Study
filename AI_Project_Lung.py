import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

training_epoch = 30
wide = 500
height = 500

#정상 폐 이미지 해상도 통일 및 파일 이름 변경


#컨볼루젼 뉴럴넷 구성
Y = tf.placeholder(tf.float32, [None, 3])

L1 = tf.layers.conv2d(X, 1024, [3, 3])
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, is_training)

L2 = tf.layers.conv2d(L1, 2048, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)

L3 = tf.layers.conv2d(L2, 4096, [3, 3])
L3 = tf.layers.max_pooling2d(L3, [2, 2], [2, 2])
L3 = tf.layers.dropout(L3, 0.7, is_training)

L4 = tf.contrib.layers.flatten(L3)
L4 = tf.layers.dense(L4, 2048, activation = tf.nn.relu)
L4 = tf.layers.dropout(L4, 0.5, is_training)

L5 = tf.contrib.layers.flatten(L4)
L5 = tf.layers.dense(L5, 256, activation = tf.nn.relu)
L5 = tf.layers.dropout(L5, 0.5, is_training)

model = tf.layers.dense(L5, 3, activation = None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels=Y))
optimizer = tf.train.RMSPropOptimizer(0.001, 0.8).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100

for epoch in range(training_epoch):
    avg_cost_list = np.zeros(len(models))
    total_batch = int()
    for j in range():