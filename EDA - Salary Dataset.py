#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


dataset = pd.read_csv('Salary.csv')
dataset.head()


# In[30]:


x = dataset.iloc[:,:1].values


# In[31]:


y = dataset.iloc[:,1:].values


# In[32]:


sns.scatterplot(x ='YearsExperience' ,y = 'Salary', data = dataset)


# In[33]:


dataset.info()


# In[36]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


regressor = LinearRegression()


# In[40]:


regressor.fit(x_train, y_train)


# In[43]:


y_pred = regressor.predict(x_test)
y_pred


# In[45]:


y_test


# In[48]:


plt.scatter(x,y)
plt.plot(x, regressor.predict(x))


# In[51]:


from sklearn.preprocessing import PolynomialFeatures


# In[53]:


poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x)


# In[54]:


regressor.fit(x_poly,y)


# In[56]:


plt.scatter(x,y)
plt.plot(x, regressor.predict(x_poly))


# In[59]:


y_pred = regressor.predict(poly.fit_transform(x))
y_pred


# # End of EDA

# In[ ]:




