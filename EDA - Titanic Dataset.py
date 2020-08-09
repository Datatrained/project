#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


train = pd.read_csv('titanic_train.csv')
train.head()


# In[10]:


# Finding missing data
train.isnull()


# In[19]:


sns.heatmap(train.isnull())


# In[18]:


sns.countplot(x = 'Survived', data = train)


# In[27]:


sns.countplot(x = 'Survived', hue = 'Sex' ,data = train)


# In[28]:


sns.countplot(x='Survived', hue = 'Pclass', data = train)


# In[38]:


sns.distplot(train['Age'], bins = 40)


# In[39]:


sns.countplot(data=train, x='SibSp')


# In[48]:


sns.distplot(train['Fare'])


# In[54]:


train['Fare'].hist()


# In[63]:


sns.boxplot(x = 'Pclass', y='Age', data = train)


# In[66]:


# Replacing NaN values with the mean of Pclass in age using def function
def inpute_age(cols):
    Age = cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        
        elif Pclass ==2:
            return 29
        
        else:
            return 24
    
    else:
        return Age


# In[67]:


train['Age'] = train[['Age', 'Pclass']].apply(inpute_age,axis=1)


# In[69]:


sns.heatmap(train.isnull())


# In[70]:


# Dropping the column 'cabin'
train.drop('Cabin', axis = 1, inplace = True)


# In[71]:


train.head()


# In[72]:


train.info()


# In[73]:


train.dropna(inplace=True)


# In[74]:


train.info()


# In[75]:


pd.get_dummies(train['Embarked'], drop_first=True).head()


# In[79]:


sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)


# In[81]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)


# In[82]:


train.head()


# In[84]:


print(embark.head())
print(sex.head())


# In[85]:


train = pd.concat([train, sex, embark], axis = 1)


# In[86]:


train.head()


# # Buildig a Logistic Regression Model

# In[87]:


train.drop('Survived', axis = 1).head()


# In[89]:


train.head()


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis = 1), train['Survived'], test_size = 0.30, random_state = 101) 


# # Training & Predecting

# In[92]:


from sklearn.linear_model import LogisticRegression


# In[95]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[114]:


predictions = logmodel.predict(X_test)
predictions


# In[115]:


from sklearn.metrics import confusion_matrix


# In[116]:


accuracy = confusion_matrix(y_test, predictions)
accuracy


# In[117]:


from sklearn.metrics import accuracy_score


# In[118]:


accuracy = accuracy_score(y_test, predictions)
accuracy


# # End of EDA
