#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# #### Clustering- Divide the universities in to group(Cluster)

# In[3]:


univ=pd.read_csv("Universities.csv")
univ


# In[4]:


univ.info()


# In[5]:


univ.isnull().sum()


# In[6]:


univ.describe()


# In[7]:


univ.boxplot()


# In[8]:


import seaborn as sns
sns.kdeplot(univ=["SAT"], fill=True, color="red")
plt.show()


# # Standardization of the data

# In[10]:


# Read all numeric columns in to univ1
Univ1=univ.iloc[:,1:]
Univ1


# In[11]:


cols=Univ1.columns
cols


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df=pd.DataFrame(scaler.fit_transform(Univ1),columns=cols)
scaled_Univ_df


# In[13]:


# Build 3 clusters using KMeans cluster algorithm
from sklearn.cluster import KMeans 
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)
                    


# In[14]:


#print the cluster labels
clusters_new.labels_


# In[15]:


set(clusters_new.labels_)


# In[16]:


#Assign clusters to the Univ data set
univ['clusterid_new'] = clusters_new.labels_


# In[17]:


univ


# In[18]:


univ.sort_values(by = "clusterid_new")


# In[38]:


# Use groupby() to find aggregated (mean) values in each cluster
univ.iloc[:,1:].groupby("clusterid_new").mean()


# # Observations:
# - The top rated cluster is clusterid_new(2).
# - The second top rated cluster is clusterid_new(1).
# - The last top rated cluster is cluster(0).

# In[44]:


univ[univ['clusterid_new']==0]


# In[48]:


wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
    


# # Observations
# - From the above graph we can choose k= 3 0r 4 which indicates elbow joins,i.e the rate of change of folw decrease.

# # clustering methods
# - Hierarchial clustering
# - Kmeans clustering
# - Kmedoids clustering
# - K-prototypes clustering
# - DBSCAN clustering

# In[ ]:




