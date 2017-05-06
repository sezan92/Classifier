#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:10:52 2017

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

#knn with gridsearch
def ClassiferSelect(X,y):
    
    knn = KNeighborsClassifier()
    k_range = list(range(1,31))
    leaf_range = list(range(1,40))
    weight_options = ['uniform', 'distance']
    algorithm_options =  ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_gridKnn = dict(n_neighbors = k_range,
                         weights = weight_options,
                         algorithm = algorithm_options
                         #leaf_size = leaf_range
                         )
    gridKNN = GridSearchCV(knn,param_gridKnn,cv=10,
                           scoring = 'accuracy') 
    gridKNN.fit(X,y)
    print "Knn Score "+ str(gridKNN.best_score_)
    print "Knn  best Params "+str(gridKNN.best_params_)
    #LogReg with gridSearch
    
    logreg = LogisticRegression()
    penalty_options =['l1','l2']
    solver_options = ['liblinear','newton_cg','lbfgs','sag']
    tol_options = [0.0001,0.00001,0.000001,0.000001]
    param_gridLog = dict(penalty=penalty_options,
                         tol=tol_options)
    gridLog = GridSearchCV(logreg,param_gridLog,cv=10,scoring='accuracy')
    gridLog.fit(X,y)
    
    print "LogReg Score "+ str(gridLog.best_score_)
    print "LogReg  best Params "+str(gridLog.best_params_)
    #NN with gridSearch
    
    NN = MLPClassifier(hidden_layer_sizes=  (8,5,4))
    activation_options = ['identity', 'logistic', 'tanh', 'relu']
    solver_options =['lbfgs', 'sgd', 'adam']
    learning_rate_options = ['constant', 'invscaling', 'adaptive']
    param_gridNN = dict(activation=activation_options,
                        solver=solver_options,
                        learning_rate = learning_rate_options)
    gridNN = GridSearchCV(NN,param_gridNN,cv=10,
                          scoring = 'accuracy')
    gridNN.fit(X,y)
    print "NN Score "+ str(gridNN.best_score_)
    print "NN  best Params "+str(gridNN.best_params_)
    
    #SVM with SVC
    flag = True
    if flag is True:
        svm = NuSVC()
        kernel_options = [ 'linear', 'sigmoid', 'rbf','precomputed']
        nu_options = np.arange(0.1,1,0.1)
        param_gridSVM = dict(kernel = kernel_options ,nu=nu_options)
        gridSVM = GridSearchCV(svm,param_gridSVM,cv=10,
                               scoring = 'accuracy')
        gridSVM.fit(X,y)
        print "SVM Score "+str(gridSVM.best_score_)
        print "SVM Params"+str(gridSVM.best_params)
        
        #Random Forest
        dtree = DecisionTreeClassifier(random_state=0)
        criterion_options = ['gini','entropy']
        splitter_options =['best','random']
        
        param_gridDtree = dict(criterion =criterion_options,splitter=splitter_options)
        
        gridDtree = GridSearchCV(dtree,param_gridDtree,cv=10,scoring='accuracy')
        gridDtree.fit(X,y)
        
        print "Decision Tree Score "+str(gridDtree.best_score_)
        print "Decision Tree params "+str(gridDtree.best_params_)
        
        #Random Forest Classifier with GridSearch
        random = RandomForestClassifier()
        n_estimators_range = list(range(1,31))
        criterion_options = ['gini','entropy']
        max_features_options =['auto','log2', None]
        param_grid = dict(n_estimators =n_estimators_range,
                          criterion= criterion_options,
                          max_features =max_features_options)
        gridRandom = GridSearchCV(random,param_grid,cv=10,
                                  scoring='accuracy')
        gridRandom.fit(X,y)
        
        print "RTrees Score "+str(gridRandom.best_score_)
        print "RTrees Best Params " +str(gridRandom.best_params_)
        
        