#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[14]:


# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv(r"C:\Users\Parth Bhutani\OneDrive\Desktop\gld_price_data.csv")


# In[15]:


# print first 5 rows in the dataframe
gold_data.head()


# In[16]:


# number of rows and columns
gold_data.shape


# In[20]:


# getting some basic informations about the data
gold_data.info()


# In[21]:


# checking the number of missing values
gold_data.isnull().sum()


# In[22]:


# getting the statistical measures of the data
gold_data.describe()


# In[23]:


correlation = gold_data.corr()


# In[24]:


# constructing a heatmap to understand the correlatiom
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':15}, cmap='hot')


# In[25]:


# correlation values of GLD
print(correlation['GLD'])


# In[26]:


# checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'],color='violet')


# In[27]:


X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[28]:


print(X)


# In[29]:


print(Y)


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# In[44]:


regressor = RandomForestRegressor(n_estimators=100)


# In[45]:


# training the model
regressor.fit(X_train,Y_train)


# In[46]:


test_data_prediction = regressor.predict(X_test)


# In[47]:


print(test_data_prediction)


# In[49]:


#R squared error 
# error score calculation on the actual data i.e Y_test and predicted data - test_data_prediction
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error: ", error_score) 


# In[51]:


#real good error as compared to the values 
#Comparison of actual and predicted values
Y_test = list(Y_test)


# In[68]:


plt.plot(Y_test, color = 'red', label = 'Actual Value')
plt.plot(test_data_prediction, color='black', label= 'Predicted Value')
plt.title('Actual Price VS Predicted Price ')
plt.xlabel('Number of Values ')
plt.ylabel('Gold Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




