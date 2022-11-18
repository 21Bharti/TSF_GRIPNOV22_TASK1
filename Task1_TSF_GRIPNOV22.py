#!/usr/bin/env python
# coding: utf-8

# ## The Sparks Foundation - GRIP - Data Science and Business Analytics - NOV'2022

# ## TASK 1: Prediction using Supervised ML

# #### Author : Bharti Rani

# #### Problem Statements: 
#     * Predict the percentage of an student based on the no. of study hours.
#     * What will be predicted score if a student studies for 9.25 hrs/ day?
# 

# #### Dataset: Student Scores
#     * It can be downloaded through link: http://bit.ly/w-data

# #### importing all the libraries

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# #### Reading the dataset

# In[2]:


df = pd.read_csv("student_scores.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# ### Visualization

# In[7]:


#Hours Vs Percentage of Scores
plt.scatter(df['Hours'], df['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Studied Hours')
plt.ylabel('Scores')
plt.show()


# In[8]:


sns.regplot(x= df['Hours'], y= df['Scores'])


# ### Train-Test Split

# In[9]:


#X will take all the values except for the last column which is our dependent variable (target variable)
X = df.iloc[:, :-1]
y = df.iloc[:,-1]


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# #### Model Building

# In[11]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()                                           


# In[16]:


regressor.fit(X_train, y_train)                                               # train the model


# In[17]:


pred_y= regressor.predict(X_test)                                             # prediction


# In[18]:


pd.DataFrame({'Actual': y_test, 'Predicted': pred_y })      # view actual and predicted on test set side-by-side


# In[24]:


# Actual vs Predicted distribution plot

sns.kdeplot(pred_y, label='Predicted', shade= True)

sns.kdeplot(y_test, label= 'Actual', shade=True)


# In[20]:


print('Train accuracy:', regressor.score(X_train, y_train), '\nTest accuracy: ', regressor.score(X_test, y_test))


# #### Predict percent for custom input value for hours
# * Ques. What will be predicted score if a student studies for 9.25 hrs/day?

# In[21]:


h= [[9.25]]
s= regressor.predict(h)
print('A student who studies', h[0][0], 'Hours is estimated to score', s[0])


# ### Conclusion:
# * We used a Linear Regression Model to predict the score of a student if he/she
# studies for 9.25 hours/day and the Predicted Score came out to be 92.91.
