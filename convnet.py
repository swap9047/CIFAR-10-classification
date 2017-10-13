# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 12:14:44 2017

@author: rai16
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
os.chdir(r'C:\Users\rai16\OneDrive - purdue.edu\CIFAR 10')

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_file(file):
    data_batch=unpickle(file)
    data_array=data_batch[b'data']
    data_labels=np.array(data_batch[b'labels']).reshape([-1,1])
    enc = OneHotEncoder(sparse=False)
    data_labels=enc.fit_transform(data_labels)
    return data_array,data_labels

def get_smallbatch(data_array,data_labels,k):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=0)
    for train_index, test_index in sss.split(data_array, data_labels):
        X_train = data_array[train_index,:]
        y_train = data_labels[train_index,:]
    return X_train,y_train

raw_images,cls=get_file('data_batch_1')
test_image,test_cls=get_file('test_batch')

#data=unpickle('data_batch_1')
#data.keys()
#data.values()
x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

####Architecture

x_image = tf.reshape(x, [-1, 3, 32, 32])##reshaping input image to 4d tensor
x_image = tf.transpose(x_image,perm=[0, 2, 3, 1])
x_image=tf.cast(x_image,dtype=tf.float32) ##casting to float32







##Weight intialisation
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

##First convolutional layer
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#First conv layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#Second conv layer
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

##Fully Connected Layer

W_fc1=weight_variable([8*8*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)



##dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


##readout layer

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2



##train and evaluation
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
batch=list()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        batch=get_smallbatch(raw_images,cls,0.01)
        if i%50==0:
            train_accuracy=accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={x: test_image, y_: test_cls, keep_prob: 1.0}))
    save_path = saver.save(sess, "my_model_final.ckpt")