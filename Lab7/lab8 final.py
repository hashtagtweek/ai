#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans 
import sklearn.metrics as sm 
import pandas as pd 
import numpy as np 


# In[2]:


iris = datasets.load_iris() 
X_train, X_test,y_train,y_test=train_test_split(iris.data, iris.target)



# In[3]:


X = pd.DataFrame(iris.data) 
print(X)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target) 
print(y)
y.columns = ['Targets'] 


# In[4]:


plt.figure(figsize=(14,7))   


# In[5]:


colormap = np.array(['red', 'lime', 'black'])   


# In[6]:


plt.subplot(1, 2, 1) 
plt.scatter(X.Sepal_Length,X.Sepal_Width, c=colormap[y.Targets], s=40) 
plt.title('Sepal')


# In[7]:


plt.subplot(1, 2, 2) 
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[y.Targets], s=40) 
plt.title('Petal')


# In[8]:


model = KMeans(n_clusters=3) 
model.fit(X) 
# This is what KMeans thought 
model.labels_ 
#print(model.labels_)
# View the results 
# Set the size of the plot 
plt.figure(figsize=(14,7))   
# Create a colormap 
colormap = np.array(['red', 'lime', 'black'])   
# Plot the Original Classifications 
plt.subplot(1, 2, 1) 
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40) 
plt.title('Real Classification')   

# Plot the Models Classifications 
plt.subplot(1, 2, 2) 
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40 ) 
plt.title('K Mean Classification')

# The fix, we convert all the 1s to 0s and 0s to 1s. 
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64) 
print(predY)

#Re-plot
 
plt.figure(figsize=(14,7))   
colormap = np.array(['red', 'lime', 'black'])   
# Plot Orginal
plt.subplot(1, 2, 1) 
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40) 
plt.title('Real Classification')   
 
 # Plot Predicted with corrected values 
plt.subplot(1, 2, 2) 
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[predY], s=40) 
plt.title('K Mean Classification')

#performance measure (accuracy)

sm.accuracy_score(y, model.labels_)
sm.confusion_matrix(y, model.labels_)


from sklearn import preprocessing 
 
scaler = preprocessing.StandardScaler() 
 
scaler.fit(X) 
xsa = scaler.transform(X) 
xs = pd.DataFrame(xsa, columns = X.columns) 
xs.sample(5) 
from sklearn.mixture import GaussianMixture 
gmm = GaussianMixture(n_components=3) 
gmm.fit(xs) 
y_cluster_gmm = gmm.predict(xs) 
y_cluster_gmm 

plt.subplot(1, 2, 1) 
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40 ) 
plt.title('GMM Classification')

sm.accuracy_score(y, y_cluster_gmm)
sm.confusion_matrix(y, y_cluster_gmm)


# In[9]:


model= KMeans(n_clusters=3)
model.fit(X_train,y_train)


# In[11]:


model.score
sm.accuracy_score(y_test, model.predict(X_test))


# In[12]:


from sklearn.mixture import GaussianMixture 


# In[10]:


model2= GaussianMixture(n_components=3)
model2.fit(X_train,y_train)


# In[11]:


model2.score
sm.accuracy_score(y_test, model2.predict(X_test))


# In[ ]:




