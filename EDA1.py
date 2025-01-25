#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.xls")
print(data)


# In[3]:


data.info()


# In[4]:


print


# In[5]:


print(type(data))
print(data.shape)
print(data.size)


# In[6]:


data1=data.drop(['Unnamed: 0',"Temp C"],axis=1)
data1


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1[data1.duplicated(keep=False)]


# In[9]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[10]:


data1.rename({'Solar.R':'Solar',},axis=1, inplace = True)
data1


# #### impute the missing values

# In[12]:


## Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[13]:


# visualize data1 missing values using heat map

cols = data1.columns
colors =['pink','black']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


# find the mean and median values of each numeric
#imputation of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[15]:


#replace the Ozone missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ",median_solar)
print("Mean of Solar: ",mean_solar)


# In[ ]:




