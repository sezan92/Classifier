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

Auto = '/home/sezan92/CifarML/Normal/Automobile'
Cat = '/home/sezan92/CifarML/Normal/Cat'
Deer = '/home/sezan92/CifarML/Normal/Deer'
Dog = '/home/sezan92/CifarML/Normal/Dog'
Horse = '/home/sezan92/CifarML/Normal/Horse'
Test ='/home/sezan92/CifarML/Normal/Test'
trainData = []
responseData = []
NumberList = []

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
        img = cv2.resize(img,(50,50))
        NumberList.append(img)    
        feature = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        trainData.append(feature.T)
        responseData.append(Label)

ReadImages(AutoImages,Auto,1)
ReadImages(CatImages,Cat,2)
ReadImages(DogImages,Dog,3)
ReadImages(DeerImages,Deer,4)
ReadImages(HorseImages,Horse,5)


X = np.float32(trainData)
y= np.float32(responseData)
#Real Stuff  Classifier Training

ClassifierSelect(X,y,num_labels=5,SVMFlag=True)    
