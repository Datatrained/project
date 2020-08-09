#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


# Loading dataset
from sklearn.datasets import load_wine


# In[18]:


wine = load_wine()


# In[19]:


wine


# In[20]:


wine.feature_names


# In[21]:


wine.target_names


# In[22]:


print(wine.DESCR)


# In[23]:


# Converting the data into a proper tabular format using Pandas
df = pd.DataFrame(wine.data, columns = wine.feature_names)
df.head()


# In[24]:


df.tail()


# In[25]:


df.shape


# In[26]:


# Adding 'Target' to the DataFrame
df['Target'] = wine.target


# In[27]:


df.head()


# In[28]:


df.tail()


# In[29]:


df.shape


# In[30]:


df.columns


# In[31]:


df.dtypes


# In[32]:


df.isnull().sum()


# In[33]:


df['Target'].value_counts()


# In[34]:


sns.countplot(df['Target'])


# In[35]:


# Plotting correlation of other features with the 'Target'
plt.figure(figsize = (12,6))
sns.heatmap(df.corr(), annot = True)


# In[36]:


sns.pairplot(df)


# In[37]:


df.hist(figsize = (12,12))


# In[38]:


# Box plots
df.plot(kind = 'box',subplots = True,  figsize = (12,12), layout = (4,4))


# In[39]:


sns.distplot(df['Target'])


# In[40]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'Target', y = 'flavanoids', data = df)


# In[41]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'Target', y = 'alcohol', data = df)


# In[42]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'Target', y = 'ash', data = df)


# In[43]:


plt.figure(figsize = (12,6))
sns.boxplot(x = 'Target', y = 'ash', data = df)


# In[44]:


sns.jointplot(x = 'Target', y = 'total_phenols', data = df, kind = 'hex')


# In[45]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'Target', y = 'alcalinity_of_ash', data = df)


# In[46]:


sns.barplot(x = 'Target', y = 'malic_acid', data = df)


# In[47]:


sns.barplot(x = 'Target', y = 'magnesium', data = df)


# In[48]:


sns.barplot(x = 'Target', y = 'hue', data = df)


# In[49]:


sns.jointplot(x = 'Target', y = 'hue', data = df, kind = 'kde')


# In[50]:


sns.barplot(x = 'Target', y = 'color_intensity', data = df)


# # Building the Model

# In[119]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.simplefilter("ignore")


# In[120]:


# Split out validation dataset
array = df.values
X = array[:, 0:13]
Y = array[:, 13]
validation_size = .33
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[121]:


print(Y)


# In[122]:


# Printing shapes of the train_test_split
print(Y.shape)
print(X.shape)


# In[123]:


print(X_train.shape)
print(X_validation.shape)
print(Y_train.shape)
print(Y_validation.shape)


# In[124]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits = 10, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    
    msg = '%s:, %f, (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[133]:


fig = plt.figure()
plt.boxplot(results)
fig.suptitle('Algorithm Comparision')
ax = fig.add_subplot(111)
ax.set_xticklabels(names)
plt.show()


# In[134]:


# Since, LDA is performing better, we shall go with LDA


# # Making predictions on the dataset

# In[137]:


LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)
predictions = LDA.predict(X_validation)


# In[140]:


print(accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))


# # End of EDA

# In[ ]:




