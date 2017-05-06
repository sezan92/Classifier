#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:05:20 2017

@author: sezan92
"""

from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
import tensorflow as tf

#Functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def xbatchCreate(num_batch,trainNp):
    batch = []
    while num_batch in range(len(trainNp)):
        batch.append(trainNp[num_batch])
        num_batch= num_batch+120
    return np.float32(batch)

def ybatchCreate(num_batch,responseNp):
    batch = []
    while num_batch in range(len(responseNp)):
        batch.append(responseNp[num_batch])
        num_batch = num_batch+120
    return np.float32(batch)

Auto = '/home/sezan92/CifarML/Deep/Automobile'
Cat = '/home/sezan92/CifarML/Deep/Cat'
Deer = '/home/sezan92/CifarML/Deep/deer'
Dog = '/home/sezan92/CifarML/Deep/Dog'
Horse = '/home/sezan92/CifarML/Deep/Horse'
Test ='/home/sezan92/CifarML/Normal/Test'
trainData = []
responseData = []

AutoImages = [ f for f in listdir(Auto) if isfile(join(Auto,f)) ]
CatImages = [ f for f in listdir(Cat) if isfile(join(Cat,f)) ]
DogImages = [ f for f in listdir(Dog) if isfile(join(Dog,f)) ]
DeerImages = [ f for f in listdir(Deer) if isfile(join(Deer,f)) ]
HorseImages = [ f for f in listdir(Horse) if isfile(join(Horse,f)) ]
TestImages = [ f for f in listdir(Test) if isfile(join(Test,f)) ]

def ReadImages(ListName,FolderName,Label):
    global NumberList
    global responseData
    global trainData
    global hog
    global cv2
    global imutils
    global winSize
    #ListName= ListName[0:100]
    for image in ListName:
        img = cv2.imread(join(FolderName,image)) 
        trainData.append(img.flatten())
        responseData.append(Label)

ReadImages(AutoImages,Auto,1)
ReadImages(CatImages,Cat,2)
ReadImages(DogImages,Dog,3)
ReadImages(DeerImages,Deer,4)
ReadImages(HorseImages,Horse,5)

trainNp = np.float32(trainData)
responseNp = np.float32(responseData)
responseNpOH = np.zeros((responseNp.shape[0],responseData[-1]))       
k=0
#OneHot
for i in range(responseNpOH.shape[0]):
    np.put(responseNpOH[i],responseNp[i]-1,1)


all_data = np.concatenate((trainNp,responseNpOH),axis=1) 
np.random.shuffle(all_data)

sess = tf.InteractiveSession()

    
x = tf.placeholder(tf.float32, shape=[None,3072])
y_ = tf.placeholder(tf.float32, shape=[None, 5])    
x_image = tf.reshape(x, [-1,32,32,3])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
    
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([4*4 * 128, 3072])
b_fc1 = bias_variable([3072])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([3072, 5])
b_fc2 = bias_variable([5])
    
    
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

num_batch =0

while num_batch<45000:
  #batch = mnist.train.next_batch(50)
  if num_batch%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:all_data[num_batch:num_batch+100,0:3072], y_:all_data[num_batch:num_batch+100,3072:3077],
                  keep_prob:1.0})
    print("step %d, training accuracy %g"%(num_batch, train_accuracy))
  train_step.run(feed_dict={x:all_data[num_batch:num_batch+100,0:3072], y_:all_data[num_batch:num_batch+100,3072:3077], keep_prob: 0.5})
  num_batch = num_batch+100


