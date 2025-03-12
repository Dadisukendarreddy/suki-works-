#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in Linear simple linear regression
# 
# 1. **Linearity:** The relationship between the predictors (X) and the response variable (Y) is linear.
# 
# 2. **Independence:** Observations are independent of each other.
# 
# 3. **Homoscedasticity:** The residuals (Y - Y_hat) exhibit constant variance at all levels of the predictor.
# 
# 4. **Normal Distribution of Errors:** The residuals (errors) of the model are normally distributed.

# ### Simple Linear Regression
# Simple linear regression is a statistical method used to model the relationship between two quantitative variables. It is one of the most basic and commonly used types of predictive analysis. The two main components in simple linear regression are:
# 
# 1. **Independent Variable (Predictor):** This is the variable that you use to make predictions. It is assumed to be the cause or the input which influences the dependent variable.
# 
# 2. **Dependent Variable (Outcome):** This is the variable that you try to predict or explain. It is assumed to be affected by the independent variable.
# 
# The goal of simple linear regression is to find a linear relationship between these two variables. This relationship is often represented as a straight line, known as the regression line, which can be written in the basic form:
# $$
# Y = \beta_0 + \beta_1X + \epsilon \
# $$
# Here:
# - 𝛽0 is the dependent variable and is the intercept of the line on the Y-axis.
# - 𝛽1 is the independent variable is the slope of the line
# 
# - 𝜖  represents the error term, accounting for the variability in \( Y \) that cannot be explained by the linear relationship with \(X\).
# 
# ### Steps in Simple Linear Regression
# 
# **1. Model Estimation:**
#    - You estimate the coefficients 𝛽0 and 𝛽1  using the method of least squares, which minimizes the sum of the squared differences (residuals) between observed values and the values predicted by the model.
# 
# **2. Prediction:**
#    - Once the model coefficients are estimated, you can use the model to predict values of \( Y \) based on new values of \( X \).
# 
# **3. Evaluation:**
#    - Assess the model’s effectiveness using metrics like R-squared, which measures the proportion of the variability in \( Y \) that can be explained by \( X \) using your model. Higher R-squared values indicate a better fit of the model to the data.
# 
# ### Assumptions
# Simple linear regression makes several key assumptions:
# - **Linearity:** The relationship between \( X \) and \( Y \) is linear.
# - **Independence:** Observations are independent of each other.
# - **Homoscedasticity:** The variance of residual is the same for any value of \( X \).
# - **Normality:** the residuals (errors), which are the differences between the observed values
# and the predicted values from the regression model, should be normally distributed. 
# 
# 
# 
# 
# 

# ### Method of Ordinary Least Squares

# In Ordinary Least Squares (OLS) regression, the coefficients of the linear equation are derived by minimizing the sum of the squares of the residuals, which are the differences between observed values and the values predicted by the model.Here's a brief step-by-step explanation of how these coefficients are derived:
# 
# Step 1: Objective Function
# The objective of OLS is to minimize the sum of the squared residuals. The sum of the squared residuals can be defined as:
# 
# $$
# \text{Minimize } S = \sum_{i=1}^n (y_i - \hat{y}_i)^2          
# $$
# 
# SSR = Sum of Squared Residuals
# 
# Step 2: Linear Model
# $$
# \hat{y}_i = \beta_0 + \beta_1 x_i
# $$
# 
# Step 3: Substituting and Expanding
# $$
# S = \sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_i))^2
# $$
# 
# Step 4: Minimizing S
# - For 𝛽0:
# $$
# \frac{\partial S}{\partial \beta_0} = -2 \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0
# $$
# - For 𝛽1:
# $$
# \frac{\partial S}{\partial \beta_1} = -2 \sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1 x_i) = 0
# $$
# 
# Step 5: Solving the Normal Equations
# 
# - For 𝛽1: (slope) is calculated as:
# $$
# \beta_1 = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sum_{i=1}^n (x_i - \overline{x})^2}
# $$
# - For 𝛽0: (intercept) is calculated using:
# $$
# \beta_0 = \overline{y} - \beta_1 \overline{x}
# $$
# 
# 
# 

# #### Import Libraries and Data Set

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[7]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# #### EDA

# In[9]:


data1.info()


# In[10]:


data1.isnull().sum()


# In[11]:


data1.describe()


# In[12]:


# Boxplot for daily column

plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[13]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[14]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[15]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# #### Observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column vaues also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots

# #### Scatter plot and Correlation Strength

# In[18]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)  
plt.ylim(0, max(y) + 100) 
plt.show()


# In[19]:


data1["daily"].corr(data1["sunday"])


# In[20]:


data1[["daily","sunday"]].corr()


# In[21]:


data1.corr(numeric_only=True)


# #### Observations on Correlation strength
# - The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong and postive with Pearson's correlation coefficient of 0.958154

# #### Fit a Linear Regression Model

# In[24]:


# Build regression model

import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[25]:


model1.summary()


# - The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# - Therefore the intercept coefficient may not be that much significant in prediction
# - However the p-value for "daily" (beta_1) is 0.00 < 0.05
# - Therfore the beta_1 coefficient is highly significant and is contributint to prediction.

# ### model.summary( ) parameters:
# 
# In regression analysis using libraries like `statsmodels` in Python, the `model.summary()` method generates a detailed summary report of the OLS regression results. This summary contains various statistical metrics and information that help in interpreting the performance and reliability of the regression model. Here's a brief explanation of some key parameters you'll typically see in the output of `model.summary()`:
# 
# #### 1. **Model Fit Statistics**
#    - **R-squared**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). A higher R-squared value indicates a better fit of the model.
#    - **Adjusted R-squared**: Adjusted for the number of predictors in the model; it is always lower than the R-squared. It provides a more accurate measure by adjusting for the number of terms in the model.
#    - **F-statistic**: A measure that tests the overall significance of the model. It compares the variance explained by the model with the variance unexplained, determining if the coefficients are different from zero collectively.
#    - **Prob (F-statistic)**: The p-value corresponding to the F-statistic. A low value (typically <0.05) indicates that the model is statistically significant.
# 
# #### 2. **Coefficients**
#    - **coef**: The estimated values of the coefficients for the predictors.
#    - **std err**: The standard error of the estimated coefficients, indicating the level of accuracy of the coefficients.
#    - **t**: The t-statistic, which is the coefficient divided by its standard error. It tests if the coefficient is significantly different from zero.
#    - **P>|t|**: The p-value corresponding to the t-statistic. A p-value less than a chosen significance level (commonly 0.05) suggests that the corresponding coefficient is statistically significant.
# 
# #### 3. **Confidence Interval**
#    - Displays the 95% confidence interval for the coefficients, giving a range within which the true coefficient is expected to fall with 95% certainty.
# 
# #### 4. **Other Diagnostics**
#    - **Durbin-Watson**: A test statistic that checks for autocorrelation in the residuals from a regression analysis. Values close to 2 suggest there is no autocorrelation.
#    - **Omnibus/Prob(Omnibus)**: A test for the normality of the residuals. A non-significant value (high p-value) suggests that the residuals are normally distributed.
#    - **Skew**: A measure of the asymmetry of the data or the residuals.
#    - **Kurtosis**: A measure of the shape of the distribution of the residuals.
#    - **Jarque-Bera (JB)/Prob(JB)**: Another test of the normality of the residuals. Similar to the Omnibus test, a higher p-value indicates more evidence for the normality of the residuals.
#    - **Condition Number**: A measure of the sensitivity of the model's output to its input. High values might indicate multicollinearity or other numerical problems.
# 
# These parameters collectively provide comprehensive insights into the model’s performance, the significance of variables, and whether the assumptions of the regression model are being met. Adjusting your model based on these diagnostics can help improve model accuracy and the reliability of inferences drawn from the analysis.

# ### **Formula for $ \ R^2  $(Coefficient of Determination) in Linear Regression**
# 
# The $\ R^2 $ value measures the proportion of variance in the dependent variable  Y  that is explained by the independent variable X  in the regression model. 
# It is calculated as:
# 
# $
# R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
# $
# 
# where:
# 
# - $ SS_{res} $(Residual Sum of Squares):  
# $
#   SS_{res} = \sum (Y_i - \hat{Y}_i)^2
# $
#   This represents the sum of squared differences between actual $ Y_i $ and predicted $ \hat{Y}_i $ values.
# 
# - $\ SS_{tot} $ (Total Sum of Squares):  
# $
#   SS_{tot} = \sum (Y_i - \bar{Y})^2
# $
#   This represents the total variance in \( Y \), where $\bar{Y} $ is the mean of \( Y \).
# 
# ### **Interpretation:**
# - $ R^2 = 1 $ → Perfect fit (all variance explained).
# - $ R^2 = 0 $ → Model does not explain any variance.
# - $ R^2 $ close to 1 → Good model fit.
# - $ R^2 $ close to 0 → Poor model fit.
# 

# In[29]:


# Plot the scatter plot and overlay the fitted straight line using matplotlib
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


# In[30]:


# Plot the linear regression line using seaborn regplot() method
sns.regplot(x="daily", y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# In[31]:


# Print the fitted line Coefficients (Beta-0 and Beta-1)
model1.params


# In[32]:


# Print the model statistics (t and p-Values)  
print(f'model t-values:\n{model1.tvalues}\n-----------------\nmodel p-values: \n{model1.pvalues}')    


# In[33]:


# Print the Quality of fitted line (R squared values)
(model1.rsquared,model1.rsquared_adj)


# #### Predict for new data points(test data)

# In[35]:


#Predict sunday sales for 200 and 300 and 1500 daily circulation
newdata=pd.Series([200,300,1500])


# In[36]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[37]:


model1.predict(data_pred)


# In[38]:


# Predict on all given training data

pred = model1.predict(data1["daily"])
pred


# In[39]:


# Add predicted values  as a column in data1
data1["Y_hat"] = pred
data1


# In[40]:


# Compute the error values (residuals) and add as another column
data1["residuals"]= data1["sunday"]-data1["Y_hat"]
data1


# ### Performance Metrics:
# 
# Linear regression models are evaluated using various performance metrics to assess the accuracy and efficiency of the model in capturing the relationship between the dependent and independent variables. 
# 
# 1. **Mean Absolute Error (MAE)**: This metric measures the average magnitude of the errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
#   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n \left| y_i - \hat{y}_i \right|$$
# 
# 
# 2. **Mean Squared Error (MSE)**: This metric measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
#    $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
# 
# 
# 3. **Root Mean Squared Error (RMSE)**: This is the square root of the mean of the squared errors. RMSE is a good measure of how accurately the model predicts the response, and it is the most important criterion for fit if the main purpose of the model is prediction.
#    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
# 
# 4. **R-squared (Coefficient of Determination)**: This metric provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
#    $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
# 
# 
# 5. **Adjusted R-squared**: This metric adjusts the R-squared value for the number of predictors in a model. It is used to determine if the addition of new predictors enhances the model or not.
#    $$\text{Adjusted } R^2 = 1 - \left(\frac{(1-R^2)(n-1)}{n-k-1}\right)$$
# 
#    where n is the number of observations and k is the number of predictors.
#    
# 6. ** Mean Absolute Percentage Error**:This formula calculates the average of the absolute percentage differences between the observed actual outcomes and the predictions made by the model.
# $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
# 

# In[42]:


# Compute Mean Squared Error for the model

mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[43]:


# Compute Mean Absolute Error (MAE)

mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[44]:


# Compute MAPE
mape = np.mean((np.abs(data1["daily"]-data1["Y_hat"])/data1["daily"]))*100
mape


# #### Checking the model rediuals scatter plot (for homoscedasticity)

# In[46]:


# Plot the residuals versus y_hat (to check wheather residuals are independent of each other)
plt.scatter(data1["Y_hat"], data1["residuals"])


# #### Observations:
# - There appears to be no trend and the residuals are randomly palced around the zero error line
# - Hence the assupmtion of homoscedasticty (constant variance in residuals) is satisfied 
# 

# In[48]:


# Plot the Q-Q plot (to check the normality of residuals)
import statsmodels.api as sm
sm.qqplot(data1["residuals"], line='45', fit=True)
plt.show()


# In[49]:


# Plot the kde distribution for residuals
sns.histplot(data1["residuals"], kde =True)


# #### Observations:
# - The data points are seen to closely follow the reference line of normality
# - Hence the residuals are approximately normally distributed as also can be seen from the kde distribution

# #### Model Improvements

# In[52]:


data2 = pd.read_csv("NewspaperData.csv")
data2


# In[53]:


# Extract the index values of outlier record
data2[data2["daily"] > 1000]


# In[54]:


# Delete the records that are outliers and rebuild the model
data3 = data2.drop([13,18])
data3.reset_index(drop=True, inplace = True)
data3


# In[55]:


# Re Build regression model (model2)

import statsmodels.formula.api as smf
model2 = smf.ols("sunday~daily",data = data3).fit()
model2.summary()


# In[56]:


pred = model2.predict(data2["daily"])
pred


# In[ ]:





# In[57]:


# Add predicted values  as a column
data3["Y_hat"] = pred
data3


# In[58]:


# Compute the erro values (residuals) and add as another column
data3["residuals"]= data3["sunday"]-data3["Y_hat"]
data3


# In[59]:


# Compute Mean Squared Error for the model

mse = np.mean((data3["daily"]-data3["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[ ]:




