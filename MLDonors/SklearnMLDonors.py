#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 19:35:17 2017

@author: sezan92
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from DataPreprocessing import*
from MLClassifier import *
#Data Preparation
import pandas as pd
train = pd.read_csv('census.csv')
#X,y = DataPreProcess(train)

#Data Preprocess
feature_cols =   ['age',	'workclass',	'education_level',	'education-num',	'marital-status', 	'occupation',	'relationship',	'race',	'sex',	'capital-gain',	'capital-loss',	'hours-per-week', 	'native-country'
                      ]

X,y = DataPreProcess(train)
X = Skewing(X,['capital-gain','capital-loss'])
X = Normalizing(X,feature_cols)
X =np.float32(X)
y =np.float32(y)
c,r =y.shape
y=y.reshape(c,) 

ClassifierSelect(X,y)



