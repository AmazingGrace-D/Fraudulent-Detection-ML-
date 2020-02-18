# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 01:05:00 2020

@author: AMAZING-GRACE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("fradulent_train.csv")

df_copy = df.copy()

df.drop(["Unnamed: 0"], inplace = True, axis = 1)

df.set_index("id", inplace = True)

df['marital_status'] = df['marital_status'].replace('unknown', 'widow')

df['transaction time'] = pd.to_datetime(df['transaction time'])

df['date'] = df['transaction time'].dt.strftime(date_format = '%c')

df['weekday'] = df['transaction time'].dt.weekday_name
df['year'] = df['transaction time'].dt.year
df['month'] = df['transaction time'].dt.month.astype(str)

df.drop(["transaction time", "date", "occupation"], inplace = True, axis = 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['account type'] = le.fit_transform(df['account type'])
df['marital_status'] = le.fit_transform(df['marital_status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['credit card type'] = le.fit_transform(df['credit card type'])
df['account source verification'] = le.fit_transform(df['account source verification'])
df['transaction source method'] = le.fit_transform(df['transaction source method'])
df['account destination verification'] = le.fit_transform(df['account destination verification'])
df['weekday'] = le.fit_transform(df['weekday'])




dataframe = pd.concat([df[['current bank amount', 'last bank amount', 'time taken (seconds)', 'most recent bank amount', 
                         'account type', 'age', 'credit card type', 'account source verification', 
                         'transaction source method', 'account destination verification', 'fradulent', 'year']], pd.get_dummies(df['marital_status'], prefix = 'marital_status'), 
                        pd.get_dummies(df['occupation'], prefix = 'occupation', drop_first = True),
                        pd.get_dummies(df['weekday'], prefix = 'weekday', drop_first = True),
                        pd.get_dummies(df['month'], prefix = 'month', drop_first = True)], axis = 1)

X = dataframe.drop('fradulent', axis = 1)
y = dataframe['fradulent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))
