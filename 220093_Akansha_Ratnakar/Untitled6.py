#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[12]:


file_path = ('C:\\Users\\91810\\Downloads\\Iris.csv')
df = pd.read_csv( file_path)
print(df.head())


# In[13]:


df = pd.DataFrame(df)
df = df.drop('Id', axis =1)
print(df)


# In[14]:


df0 = df[df['Species'] == 'Iris-setosa']
df1 = df[df['Species'] == 'Iris-versicolor']
df2 = df[df['Species'] == 'Iris-virginica']
     


# In[15]:


mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}


df['Species_Number'] = df['Species'].map(mapping)

df = df.drop('Species', axis=1)


print(df)


# In[18]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.scatter(df0['SepalLengthCm'], df0['SepalWidthCm'], color ='green')
plt.scatter(df1['SepalLengthCm'], df1['SepalWidthCm'], color ='blue')
plt.scatter(df2['SepalLengthCm'], df2['SepalWidthCm'], color ='red')


# In[19]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['PetalLengthCm'], df0['PetalWidthCm'], color = 'Green')
plt.scatter(df1['PetalLengthCm'], df1['PetalWidthCm'], color = 'Blue')
plt.scatter(df2['PetalLengthCm'], df2['PetalWidthCm'], color = 'Red')


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
from sklearn.preprocessing import StandardScaler    




# In[27]:


scaler = StandardScaler()


# In[29]:


scaler.fit(df.drop('Species_Number', axis =1))


# In[30]:


scaled_features = scaler.transform(df.drop('Species_Number', axis=1))


# In[31]:


df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())


# In[47]:


X_train,X_test, y_train, y_test = train_test_split(scaled_features,df['Species_Number'].values,
                                                    test_size=0.30)


# In[ ]:





# In[48]:


df_xtest = pd.DataFrame(X_test,columns=df.columns[:-1])
df_xtest.head()


# In[49]:


## From Scratch


# In[68]:


from collections import Counter


# In[69]:


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()

        # handle ties by randomly selecting a label among the tied ones
        max_count = most_common[0][1]
        tied_labels = [label for label, count in most_common if count == max_count]
        prediction = np.random.choice(tied_labels)

        return prediction


# In[74]:


clf = KNN(1)
clf.fit(X_train , y_train)
pred_i = clf.predict(X_test)
print(confusion_matrix(y_test,pred_i))


# In[76]:


print(classification_report(y_test,pred_i))


# In[78]:


print(accuracy_score(y_test,pred_i))
     


# In[ ]:


## plotting K vs accuracy


# In[79]:


error_rate = []


for i in range(1, 40):
    clf = KNN(i)
    clf.fit(X_train, y_train)
    pred_i = clf.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plotting the error rate
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# In[80]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# k= 5 to 18 is good //varying with cases
clf = KNN(12)
clf.fit(X_train, y_train)
pred_i = clf.predict(X_test)


print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_i))


print("\nClassification Report:")
print(classification_report(y_test, pred_i))


print("\nAccuracy Score:", accuracy_score(y_test, pred_i))


# In[81]:


## using SK learn


# In[82]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[83]:


pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))


# In[84]:


print(classification_report(y_test,pred))


# In[85]:


print(accuracy_score(y_test,pred))


# In[86]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

error_rate = []


for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))




plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# In[87]:


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))


# In[88]:


knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=12')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))


# In[ ]:




