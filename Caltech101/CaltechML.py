# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:52:18 2016

@author: sezan1992
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG

#Importing Models
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from MLClassifier import *
#Data Preparation

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



def ReadImages(ListName,FolderName,Label):
    global NumberList
    global responseData
    global trainData
    global hog
    global cv2
    global imutils
    global winSize
    for image in ListName:
        img = cv2.imread(join(FolderName,image))
        img = cv2.resize(img,(50,50))
        NumberList.append(img)    
        feature = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        trainData.append(feature.T)
        responseData.append(Label)

ReadImages(AntImages,Ant,1)
ReadImages(BeaverImages,Beaver,2)
ReadImages(ButterflyImages,Butterfly,3)
ReadImages(DolphinImages,Dolphin,4)
ReadImages(DalmatianImages,Dalmatian,5)


X = np.float32(trainData)
y= np.float32(responseData)
#Real Stuff  Classifier Training

ClassifierSelect(X,y,num_labels=5,SVMFlag=False)    
