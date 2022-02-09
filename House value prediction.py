#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


# In[2]:


from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()


# In[3]:


print(housing_data.DESCR)


# In[4]:


df = pd.DataFrame(housing_data.data)
df.columns = housing_data.feature_names
df.head()


# In[5]:


df['MedHouseVal'] = housing_data.target
df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.hist(figsize=(15,10), bins=30, edgecolor="black")
plt.show()


# In[13]:


plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="MedHouseVal", size="MedHouseVal",
               palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", loc="upper right")
plt.title("Median housing value depending on \n their spatial location")


# In[14]:


plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Longitude", y="Latitude",
               palette="viridis", alpha=0.5)
plt.title("Same plot without setting the hue and size para")


# In[12]:


df.corr()


# In[13]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation between the attributes")
plt.show()


# In[14]:


df.corr()['MedHouseVal'].sort_values()


# In[15]:


sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'])


# In[16]:


sns.scatterplot(x=df['AveRooms'], y=df['MedHouseVal'])


# In[17]:


X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[19]:


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# In[20]:


scaler = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Linear Regression 

# In[21]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)


# In[23]:


linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_mse = mean_squared_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)
print("MAE of the linear regression model is:", linreg_mae)
print("MSE of the linear regression model is:", linreg_mse)
print("R2 score of the linear regression model is:", linreg_r2)


# ### Decision Tree 

# In[24]:


dtree = DecisionTreeRegressor()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)


# In[25]:


dtree_mae = mean_absolute_error(y_test, dtree_pred)
dtree_mse = mean_squared_error(y_test, dtree_pred)
dtree_r2 = r2_score(y_test, dtree_pred)
print("MAE of the decision tree model is:", dtree_mae)
print("MSE of the decision tree model is:", dtree_mse)
print("R2 score of the decision tree model is:", dtree_r2)


# ### Random forest 

# In[26]:


rforest = RandomForestRegressor()
rforest.fit(X_train, y_train)
rforest_pred = rforest.predict(X_test)


# In[27]:


rforest_mae = mean_absolute_error(y_test, rforest_pred)
rforest_mse = mean_squared_error(y_test, rforest_pred)
rforest_r2 = r2_score(y_test, rforest_pred)
print("MAE of the random forest model is:", rforest_mae)
print("MSE of the random forest model is:", rforest_mse)
print("R2 score of the random forest model is:", rforest_r2)


# In[30]:


data = {'MedInc':7.325, 'HouseAge':30.0, 'AveRooms':5.984, 'AveBedrms':1.0238, 
        'Population':280, 'AveOccup':2.20,'Latitude':37.88, 'Longitude':-122.23}
index = [0]
new_df = pd.DataFrame(data, index)
new_df


# In[28]:


df.columns


# In[31]:


value_pred = rforest.predict(new_df)
print("The median housing value for the new data is: ", value_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




