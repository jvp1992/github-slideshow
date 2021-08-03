# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:12:21 2021

@author: josevp
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn
import xgboost
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#%% Data Prep - Reading DataSet, Eliminating Missing Values if necessary, Dealing with imbalance, Cross-Validation
data = pd.read_csv('transactional_sample.csv', encoding = 'latin-1', sep = ',')

training_dataset, test_dataset  = sklearn.model_selection.train_test_split(data)
training_dataset = training_dataset.reset_index()
training_dataset = training_dataset.drop(columns=['index'])
new_training_dataset = training_dataset 

y = new_training_dataset['has_cbk']
Y=[]
for i in range(len(y)):
    if y[i] == True:
        Y.append(1)
    if y[i] == False:
        Y.append(0)
Y_train = pd.DataFrame(Y)

X = pd.DataFrame(columns=['merchant_id', 'user_id','transaction_date', 'transaction_amount'])
X['merchant_id'] = new_training_dataset['merchant_id']
X['user_id'] = new_training_dataset['user_id']
X['transaction_amount'] = new_training_dataset['transaction_amount']
for i in range(len(y)): 
    X['transaction_date'][i] = datetime.timestamp(pd.to_datetime(new_training_dataset['transaction_date'][i]))/10**3

smt = SMOTETomek(random_state=42)
X_train, y_train = smt.fit_resample(X, Y)

#%%Preparing Test Set
test_dataset = test_dataset.reset_index()
test_dataset = test_dataset.drop(columns=['index'])
new_test_dataset = test_dataset 
y_test = new_test_dataset['has_cbk']
Y=[]
for i in range(len(y_test)):
    if y_test[i] == True:
        Y.append(1)
    if y_test[i] == False:
        Y.append(0)
Y_test = pd.DataFrame(Y)

X_test = pd.DataFrame(columns=['merchant_id', 'user_id','transaction_date', 'transaction_amount'])
X_test['merchant_id'] = new_test_dataset['merchant_id']
X_test['user_id'] = new_test_dataset['user_id']
X_test['transaction_amount'] = new_test_dataset['transaction_amount']
for i in range(len(y_test)): 
    X_test['transaction_date'][i] = datetime.timestamp(pd.to_datetime(new_test_dataset['transaction_date'][i]))/10**3

#%%RandomForestClassifier
y_tr = pd.DataFrame(y_train)
clf = RandomForestClassifier(criterion='entropy', random_state = 0)
clf.fit(X_train, y_tr)
X_tes, y_tes = smt.fit_resample(X_test, Y_test)
print(clf.score(X_tes, y_tes))

#%%Logistic
clf2 = LogisticRegression()
clf2.fit(X_train, y_tr)
print(clf2.score(X_tes, y_tes))

#%%XGBoost
X_train = np.array(X_train)
y_train = pd.to_numeric(y_train)
clf3 = xgboost.XGBClassifier()
clf3.fit(X_train, y_train)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(clf3, X_train, y_tr, scoring='roc_auc', cv=cv, n_jobs=-1)
print(scores)
X_tes = np.array(X_tes)
print(clf3.score(X_tes, y_tes))