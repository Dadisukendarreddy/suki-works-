#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold


# In[5]:


dataframe = pd.read_csv("diabetes.csv")
dataframe


# #### **Overview of Pima Indian diabetes dataset**
#  
# -Preg	Number of times pregnant	Numeric	[0, 17] 
# 
# -Gluc	Plasma glucose concentration at 2 Hours in an oral glucose tolerance test (GTIT)	Numeric	[0, 199 ]
# 
# -BP	Diastolic Blood Pressure (mm Hg)	Numeric	[0, 1 22]
# 
# -Skin	Triceps skin fold thickness (mm)	Numeric	[0 , 99]
# 
# -Insulin	2-Hour Serum insulin (µh/ml)	Numeric	[ 0, 846]
# 
# -BMI	Body mass index [weight in kg/(Height in m)]	Numeric	 [0, 67.1]
# 
# -DPF	Diabetes pedigree function	Numeric	[0 .078, 2.42]
# 
# -Age	Age (years)	Num eric	[21, 81]
# 
# -Outcome	Binary value indicating non-diabetic /diabetic	Factor	[0,1]

# In[14]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = dataframe.iloc[:,0:8]
Y = dataframe.iloc[:,8]

kfold = StratifiedKFold(n_splits=10,random_state=3,shuffle=True)

model = RandomForestClassifier(n_estimators= 200,random_state= 20,max_depth=None)
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
print(results.mean())


# In[ ]:




