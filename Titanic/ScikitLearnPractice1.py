#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on  Apr 22 23:26:40 2017

@author: sezan92
"""
#Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from MLClassifier import*
#Data Preparation
train = pd.read_csv('http://bit.ly/kaggletrain')
test = pd.read_csv('http://bit.ly/kaggletest')
feature_cols =   ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
X= train.loc[:,feature_cols]
X_new = test.loc[:,feature_cols]
y = train.loc[:,'Survived']
X = X.replace(to_replace='male',value=1)
X = X.replace(to_replace='female',value=0)
X = X.replace(to_replace='C',value=1)
X = X.replace(to_replace='Q',value=2)
X = X.replace(to_replace='S',value=3)
X = X.replace(to_replace='nan',value=0)
X_new = X_new.replace(to_replace='male',value=1)
X_new = X_new.replace(to_replace ='female',value=0)
X_new = X_new.replace(to_replace ='C',value=1)
X_new = X_new.replace(to_replace ='Q',value=2)
X_new = X_new.replace(to_replace ='S',value=3)
X_new = X_new.replace(to_replace ='nan',value=0)
X_newNp = np.float32(X_new)

#Fitting and Saving
ClassifierSelect(X,y)







