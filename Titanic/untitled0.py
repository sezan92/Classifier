#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:28:32 2017

@author: sezan92
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#Data Preparation
train = pd.read_csv('MNISTtrain.csv')
test = pd.read_csv('MNISTtest.csv')