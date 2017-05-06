#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:43:53 2017

@author: sezan92
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def DataPreProcess(train):
    feature_cols =   ['age',	'workclass',	'education_level',	'education-num',	'marital-status', 	'occupation',	'relationship',	'race',	'sex',	'capital-gain',	'capital-loss',	'hours-per-week', 	'native-country'
                      ]
    X= train.loc[:,feature_cols]
    y = train.loc[:,['income']]
    
    #sex
    X.replace(to_replace='Male',value=1,inplace = True,regex = True)
    X.replace(to_replace='Female',value=0,inplace = True,regex = True)
        
        #Race
    X.replace(to_replace='White',value=1,inplace = True,regex = True)
    X.replace(to_replace='Black',value=2,inplace = True,regex = True)
    X.replace(to_replace='Other',value=0,inplace = True,regex = True)
    X.replace(to_replace= 'Asian-Pac-Islander',value=3,inplace = True,regex = True)
    X.replace(to_replace='Amer-Indian-Eskimo',value =4,inplace = True,regex = True)
    
    #Relationship
    X.replace(to_replace='Unmarried',value = 0,inplace = True,regex = True)
    X.replace(to_replace='Husband',value =1,inplace = True,regex = True)
    X.replace(to_replace='Wife',value =2,inplace = True,regex = True)
    X.replace(to_replace='Not-in-family',value =4,inplace = True,regex = True)
    X.replace(to_replace='Other-relative',value =5,inplace = True,regex = True)
    X.replace(to_replace= 'Own-child',value=6,inplace = True, regex = True)
    
    #Employment
    X.replace(to_replace='Private',value =1,inplace = True,regex = True)    
    X.replace(to_replace='Self-emp-not-inc',value =2,inplace = True,regex = True)
    X.replace(to_replace='Self-emp-inc',value =3,inplace = True,regex = True)
    X.replace(to_replace= 'Federal-gov',value =4,inplace = True,regex = True)
    X.replace(to_replace='Local-gov',value =5,inplace = True,regex = True)
    X.replace(to_replace= 'State-gov',value =6,inplace = True,regex = True)
    X.replace(to_replace= 'Without-pay',value =7,inplace = True,regex = True)
    X.replace(to_replace='Never-worked',value =8,inplace = True,regex = True)
    
    #Education                            
    X.replace(to_replace ='Bachelors',value =1,inplace = True,regex = True)    
    X.replace(to_replace ='Some-college',value =2,inplace = True,regex = True)
    X.replace(to_replace ='11th',value =3,inplace = True,regex = True)
    X.replace(to_replace ='HS-grad',value =4,inplace = True,regex = True)
    X.replace(to_replace ='Prof-school',value =5,inplace = True,regex = True)
    X.replace(to_replace ='Assoc-acdm',value =6,inplace = True,regex = True)
    X.replace(to_replace ='Assoc-voc',value =7,inplace = True,regex = True)
    X.replace(to_replace ='9th',value =8,inplace = True,regex = True)
    X.replace(to_replace ='7th-8th',value =9,inplace = True,regex = True)
    X.replace(to_replace ='12th',value =10,inplace = True,regex = True)
    X.replace(to_replace ='Masters',value =11,inplace = True,regex = True)
    X.replace(to_replace ='1st-4th',value =12,inplace = True,regex = True)
    X.replace(to_replace ='10th',value =13,inplace = True,regex = True)
    X.replace(to_replace ='Doctorate',value =14,inplace = True,regex = True)
    X.replace(to_replace ='5th-6th',value =15,inplace = True,regex = True)
    X.replace(to_replace ='Preschool',value =16,inplace = True,regex = True)
    
    #Occupation
    X.replace(to_replace='Tech-support',value =1,inplace = True,regex = True)   
    X.replace(to_replace='Craft-repair',value =2,inplace = True,regex = True)
    X.replace(to_replace='Other-service',value =3,inplace = True,regex = True)
    X.replace(to_replace='Sales',value =4,inplace = True,regex = True)
    X.replace(to_replace='Exec-managerial',value =5,inplace = True,regex = True)
    X.replace(to_replace='Prof-specialty',value =6,inplace = True,regex = True)
    X.replace(to_replace='Handlers-cleaners',value =7,inplace = True,regex = True)
    X.replace(to_replace='Machine-op-inspct',value =8,inplace = True,regex = True)
    X.replace(to_replace='Adm-clerical',value =9,inplace = True,regex = True)
    X.replace(to_replace='Farming-fishing',value =10,inplace = True,regex = True)
    X.replace(to_replace='Transport-moving',value =11,inplace = True,regex = True)
    X.replace(to_replace='Priv-house-serv',value =12,inplace = True,regex = True)
    X.replace(to_replace='Protective-serv',value =13,inplace = True,regex = True)
    X.replace(to_replace='Armed-Forces',value =14,inplace = True,regex = True)
    
    #Familly
    X.replace(to_replace='Married-civ-spouse',value =1,inplace = True,regex = True)
    X.replace(to_replace='Divorced',value =2,inplace = True,regex = True)
    X.replace(to_replace='Never-married',value =3,inplace = True,regex = True)
    X.replace(to_replace='Separated',value =4,inplace = True,regex = True)
    X.replace(to_replace='Widowed',value =5,inplace = True,regex = True)
    X.replace(to_replace='Married-spouse-absent',value =6,inplace = True,regex = True)
    X.replace(to_replace='Married-AF-spouse',value =7,inplace = True,regex = True)
    
    #NativeCountry
    X.replace(to_replace = 'Holand-Netherlands',value =41,inplace = True,regex = True)
    X.replace(to_replace = 'Hong',value =40,inplace = True,regex = True)
    X.replace(to_replace = 'Peru',value =39,inplace = True,regex = True)
    X.replace(to_replace = 'Trinadad&Tobago',value =38,inplace = True,regex = True)
    X.replace(to_replace = 'El-Salvador',value =37,inplace = True,regex = True)
    X.replace(to_replace = 'Yugoslavia',value =36,inplace = True,regex = True)
    X.replace(to_replace = 'Thailand',value =35,inplace = True,regex = True)        
    X.replace(to_replace = 'Scotland',value =34,inplace = True,regex = True)
    X.replace(to_replace = 'Nicaragua',value =33,inplace = True,regex = True)
    X.replace(to_replace = 'Guatemala',value =32,inplace = True,regex = True)
    X.replace(to_replace = 'Hungary',value =31,inplace = True,regex = True)
    X.replace(to_replace = 'Columbia',value =30,inplace = True,regex = True)
    X.replace(to_replace = 'Haiti',value =29,inplace = True,regex = True)
    X.replace(to_replace = 'Taiwan',value =28,inplace = True,regex = True)
    X.replace(to_replace = 'Ecuador',value =27,inplace = True,regex = True)
    X.replace(to_replace = 'Laos',value =26,inplace = True,regex = True)          
    X.replace(to_replace = 'Dominican-Republic',value =25,inplace = True,regex = True)
    X.replace(to_replace = 'France',value =24,inplace = True,regex = True)
    X.replace(to_replace = 'Ireland',value =23,inplace = True,regex = True)
    X.replace(to_replace = 'Portugal',value =22,inplace = True,regex = True)
    X.replace(to_replace = 'Mexico',value =21,inplace = True,regex = True)
    X.replace(to_replace = 'Vietnam',value =20,inplace = True,regex = True)
    X.replace(to_replace = 'Jamaica',value =19,inplace = True,regex = True)
    X.replace(to_replace = 'Poland',value =18,inplace = True,regex = True)
    X.replace(to_replace = 'Italy',value =17,inplace = True,regex = True)
    X.replace(to_replace = 'Philippines',value =16,inplace = True,regex = True)        
    X.replace(to_replace = 'Honduras',value =15,inplace = True,regex = True)
    X.replace(to_replace = 'Iran',value =14,inplace = True,regex = True)
    X.replace(to_replace = 'Cuba',value =13,inplace = True,regex = True)
    X.replace(to_replace = 'China',value =12,inplace = True,regex = True)
    X.replace(to_replace = 'South',value =11,inplace = True,regex = True)
    X.replace(to_replace = 'Greece',value =10,inplace = True,regex = True)
    
    X.replace(to_replace = 'Japan',value =9,inplace = True,regex = True)
    X.replace(to_replace = 'India',value =8,inplace = True,regex = True)
    X.replace(to_replace = 'Outlying',value =7,inplace = True, regex = True)
    X.replace(to_replace = 'Germany',value =6,inplace = True,regex = True)
    X.replace(to_replace = 'Canada',value =5,inplace = True,regex = True)
    X.replace(to_replace = 'Puerto-Rico',value =4,inplace = True,regex = True)
    X.replace(to_replace = 'England',value =3,inplace = True,regex = True)
    X.replace(to_replace = 'Cambodia',value =2,inplace = True,regex = True)
    X.replace(to_replace = 'United-States',value =1,inplace = True,regex = True)
    
    y.replace(to_replace = '<=50K',value =1,inplace = True,regex = True)
    y.replace(to_replace = '>50K',value =0,inplace = True,regex = True)
    
   
    return X,y

def Skewing(X,feature_cols):
    X[feature_cols] = X[feature_cols].apply(lambda x:np.log(x+1))
    return X

def Normalizing(X,feature_cols):
    scalar = MinMaxScaler()
    X[feature_cols] = scalar.fit_transform(X[feature_cols])
    return X