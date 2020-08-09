#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing required Libraries
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[3]:


# Reading data
dat = pd.read_csv('Sonar.csv')
dat.head()


# In[4]:


# Converting 'Class' column to 'Category' data type
X = dat.drop('Class',1)
dat['Class']=dat['Class'].astype('category')
Y = dat['Class']


# In[6]:


# Split data into train_test_data sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=4)


# In[7]:


# Fit the logistic Regression model
logReg = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, Y_train)


# In[8]:


# Evaluate the moel on test data set
Y_pred = logReg.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))


# # End of EDA
