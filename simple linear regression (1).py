#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[6]:


data1.describe()


# In[7]:


data1.boxplot()


# In[8]:


sns.kdeplot(data=data1["daily"], fill=True, color="blue")
sns.rugplot(data=data1["sunday"], color="black")
plt.show()


# In[9]:


sns.scatterplot(data=data1,x="daily",y="sunday",color="blue")
plt.title("Scatter plot of Daily vs Sunday")
plt.xlabel("Daily")
plt.ylabel("Sunday")
plt.show()


# In[38]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[40]:


plt.figure(figsize=(6,3))
plt.title("Box plot for sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[10]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[36]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# # Observations
# - There are no missing values
# - The daily column vcalues appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the 

# # Scatter plot and Correlation strength

# In[50]:


x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"],data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[13]:


data1[["daily","sunday"]].corr()


# In[54]:


data1.corr(numeric_only=True)


# # Observations on Correlation strength
# - The realtionship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong and positive with Pearson's correlation coefficient of 0.958154

# # Fit a Regression model

# In[14]:


# Build regression model

import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[15]:


model.summary()


# In[16]:


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




