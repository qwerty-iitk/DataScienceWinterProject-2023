#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[6]:





# In[121]:


df = pd.read_csv("C:/Users/91810/Downloads/Social_Network_Ads.csv")


# In[122]:


df.head()


# In[124]:


df.shape


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


unique_counts = df.nunique()
print(unique_counts)


# In[16]:


# From Scratch


# In[58]:


X_train, X_test, Y_train, Y_test = train_test_split(df['Age'], df['Purchased'], test_size=0.30)


# In[104]:


# NORMALISATION

def normalize(X):
    return X - X.mean()


# In[105]:


# STANDARDISATION

def standardize(X):
    mean_X = np.mean(X)
    std_X = np.std(X)

    standardized_X = (X - mean_X) / std_X
    return standardized_X


# In[106]:


# LOGISTIC REGRESSSION 

def predict(X, b0, b1):
    return np.array([1 / (1 + np.exp(-1*b0 + -1*b1*x)) for x in X])


# In[107]:


def logistic_regression(X, Y):
    
    m=0
    c=0
    
    lr = 0.001
    epochs = 10000

    for epoch in range(epochs):
        y_pred = predict(X, c, m)
        
        dc = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))
        dm = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))
        
        c = c - lr* dc
        m = m - lr * dm
    
    return c, m


# In[108]:


# accuracy of the model
def accuracy_model(test_data_x, test_data_y, c, m):
    accuracy = 0
    
    y_pred = predict(test_data_x, c, m)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    
    for i in range(len(y_pred)):
        if y_pred[i] == test_data_y.iloc[i]:
            accuracy += 1
    return accuracy*100 / len(y_pred)


# In[109]:


# Training 
c1, m1= logistic_regression(X_train, Y_train)
c2, m2= logistic_regression(normalize(X_train), Y_train)
c3, m3= logistic_regression(standardize(X_train), Y_train)


# In[110]:


# Accuracy Calculation
print("Accuracies of models:\n")
print("Using raw data:", round(accuracy_model(X_test, Y_test, c1, m1), 4))
print("Using normalised data:", round(accuracy_model(normalize(X_test), Y_test, c2, m2),4))
print("Using standardised data:", round(accuracy_model(standardize(X_test), Y_test, c3, m3),4))


# In[111]:


# USING SCIKIT LEARN


# In[112]:


# splitting into train and test set
x = df[['Age', 'EstimatedSalary']].values 
y = df['Purchased']

#split the datset into training and testing sets (70:30)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)


# In[113]:


# Normalisation
norm = MinMaxScaler().fit(train_x)

train_norm_x = norm.transform(train_x)
test_norm_x = norm.transform(test_x)


# In[114]:


# standardisation 
scaler = StandardScaler() 

x_train = scaler.fit_transform(train_x) 
x_test = scaler.fit_transform(test_x)


# In[115]:


#LOGISTIC REGRESSION 


# In[120]:


# For raw data
model_1 = LogisticRegression(max_iter=10000, random_state=0)

# Train the model
model_1.fit(train_x, train_y)


predictions_1 = model_1.predict(test_x)
accuracy_1 = accuracy_score(test_y, predictions_1)
print("Accuracy:", f'{accuracy_1 * 100:.2f}%')


# In[117]:


# for normalised data 
model_2 = LogisticRegression(max_iter=10000, random_state=0) 
model_2.fit(train_norm_x, train_y)

predictions_2 = model_2.predict(test_norm_x)

accuracy_2 = accuracy_score(test_y, predictions_2)

#print("Accuracy:", f'{accuracy_2*100}%')
print("Using normalised data:", f'{accuracy_2*100:.4f}%')


# In[118]:


# for standardised data
model_3 = LogisticRegression(max_iter=10000, random_state=0)

# Train the model
model_3.fit(x_train, train_y)

predictions_3 = model_3.predict(x_test)

accuracy_3 = accuracy_score(test_y, predictions_3)
print("Accuracy:", f'{accuracy_3 * 100:.2f}%')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




