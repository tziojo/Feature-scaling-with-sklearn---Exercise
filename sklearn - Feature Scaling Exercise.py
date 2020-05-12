#!/usr/bin/env python
# coding: utf-8

# # Feature scaling with sklearn - Exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. This exercise is very similar to a previous one. This time, however, **please standardize the data**.
# 
# Apart from that, please:
# -  Display the intercept and coefficient(s)
# -  Find the R-squared and Adjusted R-squared
# -  Compare the R-squared and the Adjusted R-squared
# -  Compare the R-squared of this regression and the simple linear regression where only 'size' was used
# -  Using the model make a prediction about an apartment with size 750 sq.ft. from 2009
# -  Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?
# -  Create a summary table with your findings
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size' and 'year'.
# 
# Good luck!

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# ## Load the data

# In[3]:


data=pd.read_csv('real_estate_price_size_year.csv')
data.head()


# In[4]:


data.describe()


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[5]:


x=data[['size','year']]
y=data['price']


# ### Scale the inputs

# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)


# ### Regression

# In[8]:


reg=LinearRegression()
reg.fit(x_scaled,y)


# ### Find the intercept

# In[11]:


reg.intercept_


# ### Find the coefficients

# In[13]:


reg.coef_


# ### Calculate the R-squared

# In[14]:


reg.score(x_scaled,y)


# ### Calculate the Adjusted R-squared

# In[18]:


def adj_r2(x,y):
    r2=reg.score(x,y)
    n=x.shape[0]
    p=x.shape[1]
    adj_r2=1-(1-r2)*(n-1)/(n-p-1)
    return adj_r2


# In[19]:


adj_r2(x_scaled,y)


# ### Compare the R-squared and the Adjusted R-squared

# It seems the the R-squared is only slightly larger than the Adjusted R-squared, implying that we were not penalized a lot for the inclusion of 2 independent variables.

# ### Compare the Adjusted R-squared with the R-squared of the simple linear regression

# Comparing the Adjusted R-squared with the R-squared of the simple linear regression (when only 'size' was used - a couple of lectures ago), we realize that 'Year' is not bringing too much value to the result.

# ### Making predictions
# 
# Find the predicted price of an apartment that has a size of 750 sq.ft. from 2009.

# In[21]:


new_data=[[750,2009]]
new_data_scaled = scaler.transform(new_data)


# In[22]:


reg.predict(new_data_scaled)


# ### Calculate the univariate p-values of the variables

# In[24]:


from sklearn.feature_selection import f_regression


# In[25]:


f_regression(x_scaled,y)


# In[27]:


pvalues=f_regression(x_scaled,y)[1]
pvalues


# In[28]:


pvalues.round(3)


# ### Create a summary table with your findings

# In[32]:


reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p_values'] = pvalues.round(3)
reg_summary


# It seems that 'Year' is not event significant, therefore we should remove it from the model.

# In[ ]:




