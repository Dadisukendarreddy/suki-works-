#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[3]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[7]:


data1.info()


# In[9]:


print(type(data1))
print(data1.shape)
print(data1.size)


# In[11]:


data1.describe()


# In[13]:


data1.boxplot()


# In[25]:


sns.kdeplot(data=data1["daily"], fill=True, color="blue")
sns.rugplot(data=data1["sunday"], color="black")
plt.show()


# In[31]:


sns.scatterplot(data=data1,x="daily",y="sunday",color="blue")
plt.title("Scatter plot of Daily vs Sunday")
plt.xlabel("Daily")
plt.ylabel("Sunday")
plt.show()


# In[33]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[35]:


data1["daily"].corr(data1["sunday"])


# In[39]:


data1[["daily","sunday"]].corr()


# In[43]:


data1[["daily","sunday"]].corr()


# In[49]:


# Build regression model

import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[51]:


model.summary()


# In[53]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x
 
# plotting the regression line
plt.plot(x, y_hat, color = "g")
  
# putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




