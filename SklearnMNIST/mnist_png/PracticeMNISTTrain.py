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

Zero = '/home/sezan92/SklearnMNIST/mnist_png/training/0'
One= '/home/sezan92/SklearnMNIST/mnist_png/training/1'
Two = '/home/sezan92/SklearnMNIST/mnist_png/training/2'
Three = '/home/sezan92/SklearnMNIST/mnist_png/training/3'
Four = '/home/sezan92/SklearnMNIST/mnist_png/training/4'
Five = '/home/sezan92/SklearnMNIST/mnist_png/training/5'
Six = '/home/sezan92/SklearnMNIST/mnist_png/training/6'
Seven = '/home/sezan92/SklearnMNIST/mnist_png/training/7'
Eight = '/home/sezan92/SklearnMNIST/mnist_png/training/8'
Nine = '/home/sezan92/SklearnMNIST/mnist_png/training/9'

trainData = []
responseData = []
NumberList = []
ZeroImages = [ f for f in listdir(Zero) if isfile(join(Zero,f)) ]
OneImages = [ f for f in listdir(One) if isfile(join(One,f)) ]
TwoImages = [ f for f in listdir(Two) if isfile(join(Two,f)) ]
ThreeImages = [ f for f in listdir(Three) if isfile(join(Three,f)) ]
FourImages = [ f for f in listdir(Four) if isfile(join(Four,f)) ]
FiveImages = [ f for f in listdir(Five) if isfile(join(Five,f)) ]
SixImages = [ f for f in listdir(Six) if isfile(join(Six,f)) ]
SevenImages = [ f for f in listdir(Seven) if isfile(join(Seven,f)) ]
EightImages = [ f for f in listdir(Eight) if isfile(join(Eight,f)) ]
NineImages = [ f for f in listdir(Nine) if isfile(join(Nine,f)) ]



def ReadImages(ListName,FolderName,Label):
    global NumberList
    global responseData
    global trainData
    global hog
    global cv2
    global imutils
    global winSize
    ListName= ListName[0:100]
    for image in ListName:
        img = cv2.imread(join(FolderName,image))
        NumberList.append(img)    
        feature = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        trainData.append(feature.T)
        responseData.append(Label)


ReadImages(ZeroImages,Zero,0)
ReadImages(OneImages,One,1)
ReadImages(TwoImages,Two,2)
ReadImages(ThreeImages,Three,3)
ReadImages(FourImages,Four,4)
ReadImages(FiveImages,Five,5)
ReadImages(SixImages,Six,6)
ReadImages(SevenImages,Seven,7)
ReadImages(EightImages,Eight,8)
ReadImages(NineImages,Nine,9)

X = np.float32(trainData)
y= np.float32(responseData)
#Real Stuff  Classifier Training

ClassifierSelect(X,y,num_labels=10,SVMFlag=True)    
