#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


# read the data from csv file 
cars = pd.read_csv("Cars.csv")
cars.head()


# # Description of columns 
# - MPG: Milege of the car (Mile per Gallon)
# - HP: Horse Power of the car (X1 column)
# - VOL: Volume of the car(size) (X2 column)
# - SP: Top speed of the car (Miles per Hour) (X3 column)
# - WT: Weight of the car (Pounds)(X4 coloumn)

# In[ ]:




