#!/usr/bin/env python
# coding: utf-8

# In[5]:



from sklearn import datasets 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics 



# In[6]:


iris= datasets.load_iris()
X_train, X_test,y_train,y_test=train_test_split(iris.data, iris.target)


# In[7]:


model= KMeans(n_clusters=3)
model.fit(X_train,y_train)


# In[ ]:





# In[8]:


model.score
metrics.accuracy_score(y_test, model.predict(X_test))


# In[9]:


from sklearn.mixture import GaussianMixture 


# In[10]:


model2= GaussianMixture(n_components=3)
model2.fit(X_train,y_train)


# In[11]:


model2.score
metrics.accuracy_score(y_test, model2.predict(X_test))


# In[ ]:




