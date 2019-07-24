#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('/Users/sadegh/Desktop/DataSet GitHub/Regression/50_startups.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


sns.pairplot(df)


# In[8]:


sns.distplot(df['Profit'])


# In[9]:


sns.heatmap(df.corr())


# In[10]:


X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lm = LinearRegression()


# In[15]:


lm.fit(X_train,y_train)


# In[16]:


# print the intercept
print(lm.intercept_)


# In[17]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[18]:


predictions = lm.predict(X_test)


# In[19]:


plt.scatter(y_test,predictions)


# In[20]:


sns.distplot((y_test-predictions),bins=50);


# In[21]:


from sklearn import metrics


# In[22]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




