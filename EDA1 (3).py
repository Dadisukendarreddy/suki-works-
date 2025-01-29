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


# In[17]:


print(data1["Weather"].value_counts())
mode_Weather = data1["Weather"].mode()[0]
print(mode_Weather)


# In[18]:


# impute missing values
data1["Weather"] = data1["Weather"].fillna(mode_Weather)
data1.isnull().sum()


# In[19]:


print(data1["Month"].value_counts())
mode_Month = data1["Month"].mode()[0]
print(mode_Month)


# In[20]:


data1["Month"] = data1["Month"].fillna(mode_Month)
data1.isnull().sum()


# In[21]:


print(data1["Solar"].value_counts())
mode_Solar = data1["Solar"].mode()[0]
print(mode_Solar)


# In[22]:


data1["Solar"] = data1["Solar"].fillna(mode_Solar)
data1.isnull().sum()


# # Detection of outliers in the columns

# In[24]:


#create a figure with two sublots, stacked vertically
fig, axes = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

#plot the box plot in the first (top) subplot
sns.boxplot(data=data1["Ozone"],ax=axes[0],color='blue',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

sns.histplot(data1["Ozone"],kde=True, ax=axes[1], color='red',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.show()


# # observations
# -> The ozone column has extreme values beyond 81 as seen from box plot 
# 
# -> the same is confirmed from the below right-skewed histogram

# In[26]:


fig, axes = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1,3]})

#plot the box plot in the first (top) subplot
sns.boxplot(data=data1["Solar"],ax=axes[0],color='skyblue',width=0.5,orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

sns.histplot(data1["Solar"],kde=True, ax=axes[1], color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.show()


# # observations
# -> There is no outliers in the solar
# 
# -> slightly left histogram

# In[28]:


sns.violinplot(data=data1["Ozone"],color='lightgreen')
plt.title("Violin plot")
plt.show()


# In[29]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"],vert= False)


# In[30]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[31]:


data1["Ozone"].describe()


# In[32]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# # observations
# . It is observed that only two outliers are indefined using std method
# 
# .In boxplot method more no of outliers are defined
# 
# . This is because the assumption of normality is not satisified in this column

# # Quantile-Quantile plot for detection of outliers

# In[35]:


import scipy.stats as stats


plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outliers detection", fontsize=14)
plt.xlabel("Theorectical Quantiles", fontsize=12)


# # Observations from Q-Q plot
# .The data does not follow normal distribution as the data points are deviating signficantly away from the red line
# 
# .The data shows a right-skewed distribution and possible outliers 

# In[69]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin Plot")
plt.show()


# In[73]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[75]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="green",palette="Set1",size=6,jitter=True)


# In[84]:


# kdeplot and rugplot;
sns.kdeplot(data=data1["Ozone"], fill=True, color="orange")
sns.rugplot(data=data1["Ozone"], color="black")


# In[90]:


# category wise boxplot for ozone
sns.boxplot(data = data1, x = "Weather", y= "Ozone")


# # corelation coefficient

# In[94]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[101]:


# compute pearson correlation coefficient

data1["Wind"].corr(data1["Temp"])


# In[103]:


# Read all numeric columns into a new table
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




