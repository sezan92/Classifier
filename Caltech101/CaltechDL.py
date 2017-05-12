#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 09:05:20 2017

@author: sezan92
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:52:18 2016

@author: sezan1992
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
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

def num2name(num):
    if num==1:
        name= 'Ant'
    elif num==2:
        name = 'Beaver'
    elif num==3:
        name = 'Butterfly'
    elif num==4:
        name = 'Dolphin'
    elif num==5:
        name = 'Dalmatian'
    return name
print "Preparing Data"

Ant = '/home/sezan92/Classifier/Caltech101/ant'
Beaver = '/home/sezan92/Classifier/Caltech101/beaver'
Butterfly = '/home/sezan92/Classifier/Caltech101/butterfly'
Dalmatian = '/home/sezan92/Classifier/Caltech101/dalmatian'
Dolphin = '/home/sezan92/Classifier/Caltech101/dolphin'
Test ='/home/sezan92/Classifier/Caltech101/Test'
trainData = []
responseData = []
NumberList = []

AntImages = [ f for f in listdir(Ant) if isfile(join(Ant,f)) ]
BeaverImages = [ f for f in listdir(Beaver) if isfile(join(Beaver,f)) ]
ButterflyImages = [ f for f in listdir(Butterfly) if isfile(join(Butterfly,f)) ]
DolphinImages = [ f for f in listdir(Dolphin) if isfile(join(Dolphin,f)) ]
DalmatianImages = [ f for f in listdir(Dalmatian) if isfile(join(Dalmatian,f)) ]
TestImages = [ f for f in listdir(Test) if isfile(join(Test,f)) ]

def ReadImages(ListName,FolderName,Label,size =(48,48)):
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
        img = cv2.resize(img,size)
        trainData.append(img.flatten())
        responseData.append(Label)
print "Reading Images"

ReadImages(AntImages,Ant,1)
ReadImages(BeaverImages,Beaver,2)
ReadImages(ButterflyImages,Butterfly,3)
ReadImages(DolphinImages,Dolphin,4)
ReadImages(DalmatianImages,Dalmatian,5)

Size = (48,48)
trainNp = np.float32(trainData)
responseNp = np.float32(responseData)
responseNpOH = np.zeros((responseNp.shape[0],responseData[-1]))       
k=0
#OneHot
for i in range(responseNpOH.shape[0]):
    np.put(responseNpOH[i],responseNp[i]-1,1)


all_data = np.concatenate((trainNp,responseNpOH),axis=1) 
np.random.shuffle(all_data)

print "Data Ready"

print "Starting Tensorflow"

sess = tf.InteractiveSession()

feature_cols = trainNp.shape[1] 
num_labels = responseNpOH.shape[1]

x = tf.placeholder(tf.float32, shape=[None,feature_cols])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])    
x_image = tf.reshape(x, [-1,Size[0],Size[1],3])
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

W_fc1 = weight_variable([(Size[0]/8)*(Size[1]/8)* 128, feature_cols])
b_fc1 = bias_variable([feature_cols])

h_pool3_flat = tf.reshape(h_pool3, [-1, (Size[0]/8)*(Size[1]/8)* 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([feature_cols, num_labels])
b_fc2 = bias_variable([num_labels])
    
    
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

num_batch =0

while num_batch<all_data.shape[0]:
  #batch = mnist.train.next_batch(50)
  if num_batch%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:all_data[num_batch:num_batch+10,0:feature_cols], y_:all_data[num_batch:num_batch+10,feature_cols:feature_cols+num_labels],
                  keep_prob:1.0})
    print("step %d, training accuracy %g"%(num_batch, train_accuracy))
  train_step.run(feed_dict={x:all_data[num_batch:num_batch+10,0:feature_cols], y_:all_data[num_batch:num_batch+10,feature_cols:feature_cols+num_labels], keep_prob: 0.5})
  num_batch = num_batch+10



for image in TestImages:
    img = cv2.imread(join(Test,image))
    img2 = cv2.resize(img,(48,48)).reshape((1,6912))
    testData =img2
    label = sess.run(y_conv,feed_dict={x:testData,keep_prob:1.0})
    pred = np.argmax(label)+1
    plt.figure()
    plt.imshow(img)
    plt.title(num2name(pred))
    
