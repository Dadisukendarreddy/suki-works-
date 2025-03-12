#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
import numpy as np


# In[3]:


import pandas as pd 
cars = pd.read_csv("Toyoto_Corrola.csv")
cars.head()


# In[5]:


cars.info()


# In[7]:


cars.isna().sum()


# In[9]:


cars[cars.duplicated()]


# # Observations
# - There are no null values
# - There is no missing values
# - In the above dataset [dtypes: int64(9), object(1)]
# - In the given data there are 5 continuous and 3 categorical column
# 
# 

# In[14]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='Price',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='Price',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[18]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='Age_08_04',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='Age_08_04',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[20]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='KM',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='KM',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[22]:


fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[40]:


import seaborn as sns
counts = cars["Gears"].value_counts()
plt.bar(counts.index, counts.values)
 


# In[ ]:




