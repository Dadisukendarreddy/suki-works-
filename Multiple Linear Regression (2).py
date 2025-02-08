#!/usr/bin/env python
# coding: utf-8

# #### Assumption in Multi linear Regression
# 1. Linearity: The relationship between the predictors(X) and the response (Y) is linear
# 2. Independence: Observation are independent of each other
# 3. Homoscedasticity: The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Erroe: The residuals of the model are normally distributed
# 5. No multicollinearity: The independent variable should not be too highly correlated with each other.
# - Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of columns
# - MPG: Milege of the car(Mile per Gallon)(This is T-column to be predicted)
# - HP:Horse Power of the car(X1 column)
# - VOL:Volume of the car(size)(X2 column)
# - SP: Top speed of the car(Miles per Hour)(X3 column)
# - WT:Weight of the car(pounds)(X4 column)

# #### EDA

# In[7]:


cars.info()


# In[8]:


#check for missing values.
cars.isna().sum()


# #### Observations about info(),missing values
# - There are no missing values
# - There are 81 observation (81 diffrent cars data)
# - The data type of the columns are also relevant and valid

# In[10]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='HP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[11]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='VOL',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='VOL',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='SP',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='SP',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[13]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='MPG',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='MPG',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[14]:


#create a figure with two subplots (one above the other)
fig,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={"height_ratios":(.15,.85)})
sns.boxplot(data=cars,x='WT',ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='WT',ax=ax_hist,bins=30,kde=True,stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# #### Observation from boxplot and histogram
# - These are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# - In VOL and WT columns,a few outliers are observed in both tails of their distribution
# - The extreme values of cars data may hava come from the specially designed nature of cars
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may to be considered while building the regression model

# #### Checking for duplicated rows

# In[17]:


cars[cars.duplicated()]


# #### Pairs plots and Correlation Coefficients

# In[19]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[20]:


cars.corr()


# #### Observations
# - The highest correlation coefficient is  between HP and Sp is(0.973848)
# - The next Highest correlation coefficient is between VOL and VOL(1.000000)
# - Between x and y.all the x variable are showing moderste to high correlation strength, highest being between HP and MPG.
# - Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# - Among x columns (x1,x2,x3 and x4),some very high correlation strength are observed between SP vs HP,VOL vs WT
# - The high correlation among x columns is not desirable as it might lead to multicollinearity problem 

# #### Preparing a preliminary model considering all X columns

# In[23]:


import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[24]:


model1.summary()


# #### Observation from model summary
# - The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# - The probability value with respect to F-statistic is colse to zero,undicating that all or some of X columns are signficant
# - The p-value for VOL and WT are higher than 5% indicating some interaction issue among themselves,which need to be further explored

# #### Performance metric for model1 (By Mean Squared Error)

# In[27]:


#Find the performance metrics
#create a data frame with actual y and predicated y columns
df1=pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[28]:


pred_y1=model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[29]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# Checking for multicolling among X-columns using VIF method

# In[31]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Observations for VIF values:
# - The ideal range of VIF values shall be betweeen 0 to 10. However slightly higher values can be tolerated.
# - As seen from the very high Compute VIF values for VOL and WT, it is clear that they are prone to multicollinearity problem.
# - Hence it is decided to drop one of the columns(either VOL or WT)to overcome the multicollinearity.
# - It is decided to drop WT and retain..d1)  
# Vif_frame

# In[33]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[34]:


# Bulid model2 on cars1 dataset
import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[35]:


model2.summary()


# # Performance ,metrics for model2

# In[37]:


#Find the performance metrics 
# Create a data frame with actual y and predicted y columns

df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[38]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[39]:


from sklearn.metrics import mean_squared_error
mse= mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# #### Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[67]:


# define variables and assign values 
k = 3
n = 81
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[ ]:





# In[71]:


from statsmodels.graphics.regressionplots import influence_plot

influence_plot(model1,alpha=.05)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()


# # observations
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influencers.
# - as their H Leverasge values are higher and size is higher.

# In[74]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[80]:


cars2=cars1.drop(cars1.index[[65,70,76,78,80]],axis=0).reset_index(drop=True)
cars2
                                                                  


# # Bulid Model3 on cars2 dataset

# In[83]:


#Rebuild the model 
model3= smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[90]:


model3.summary()


# # Performance Metrics for model3

# In[88]:


df3= pd.DataFrame()
df3["actual_y3"] = cars["MPG"]
df3.head()


# In[92]:


# Predict on all X data columns
pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[94]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"],df3["pred_y3"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[99]:


#plot the Q-Q plot (to check the normality of residuals)
import statsmodels.api as sm 
sm.qqplot(df3["actual_y3"],line='45', fit=True)


# In[104]:


#Plot the residuals versus y_hat (to check wheather residuals are independent of each other
plt.scatter(df3["actual_y3"],df3["pred_y3"])


# In[ ]:





# In[ ]:




