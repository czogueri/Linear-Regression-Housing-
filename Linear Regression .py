#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('Desktop/USA_Housing.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


sns.pairplot(df)


# In[10]:


sns.distplot(df['Price'])


# In[12]:


sns.heatmap(df.corr(), annot=True)


# In[13]:


df.columns


# In[14]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
      'Avg. Area Number of Bedrooms', 'Area Population']]


# In[33]:


y = df['Price']


# In[20]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=101)


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


lm = LinearRegression()


# In[35]:


lm.fit(X_train,y_train)


# In[36]:


print(lm.intercept_)


# In[32]:


lm.coef_


# In[28]:


X_train.columns


# In[37]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[38]:


cdf


# Predictions 

# In[41]:


predictions = lm.predict(X_test)


# In[42]:


predictions


# In[43]:


#actual prices
y_test


# In[44]:


plt.scatter(y_test, predictions)


# In[46]:


sns.distplot((y_test-predictions))


# In[47]:


from sklearn import metrics


# In[48]:


metrics.mean_absolute_error(y_test,predictions)


# In[50]:


metrics.mean_squared_error(y_test, predictions)


# In[51]:


np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[1]:


n = int(input('Find Primes up to:'))


# In[ ]:




