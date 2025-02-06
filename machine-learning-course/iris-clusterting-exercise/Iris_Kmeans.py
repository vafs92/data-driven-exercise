#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np


# In[112]:


df=pd.read_csv("iris.csv", sep=",")
df.head()


# In[113]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Label']=le.fit_transform(df['Label'])


# In[114]:


df.head()


# In[115]:


from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=4, random_state=0, algorithm='auto', init='k-means++', max_iter=300)


# In[116]:


y_kmeans = kmean.fit(df)


# In[117]:


kmean.cluster_centers_ #centroid


# In[118]:


kmean.n_iter_ #berapa kli iterasi


# In[119]:


kmean.labels_


# In[120]:


import seaborn as sms


# In[121]:


sms.scatterplot(data=df,x='Sepal Length', y='Sepal Width', hue=kmean.labels_)


# In[122]:


sms.scatterplot(data=df,x='Petal Length', y='Petal Width', hue=kmean.labels_)


# In[123]:


from sklearn.cluster import KMeans


# In[124]:


kmeans_kwargs = {
    "init": 'random',
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42
}

sse=[]

for k in range (1,20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
    


# In[125]:


import matplotlib.pyplot as plt


# In[126]:


#metode elbow

plt.style.use('fivethirtyeight')
plt.plot(range(1,20),sse)
plt.xticks(range(1,20))
plt.xlabel('Jumlah Klaster')
plt.ylabel('WCSS')
plt.show


# In[127]:


from kneed import KneeLocator


# In[128]:


km =KneeLocator(range(1,20),sse,curve="convex", direction = "decreasing")


# In[129]:


km.elbow


# In[130]:


from sklearn.metrics import silhouette_score


# In[131]:


#silhouette_cofficient
sc = []

for k in range (2,20):
    kmean = KMeans(n_clusters=k, **kmeans_kwargs)
    kmean.fit(df)
    
    score = silhouette_score(df, kmean.labels_)
    sc.append(score)


# In[132]:


plt.style.use('fivethirtyeight')
plt.plot(range(2,20),sc)
plt.xticks(range(2,20))
plt.xlabel('Jumlah Klaster')
plt.ylabel('SC')
plt.show


# In[133]:


#TARGET DIHILANGKAN


# In[134]:


df_new=pd.read_csv("iris.csv", sep=",")
df_new.head()


# In[135]:


df_new.drop("Label",axis=1,inplace=True)
df_new.head()


# In[136]:


kmean_new = KMeans(n_clusters=4, random_state=0, algorithm='auto', init='k-means++', max_iter=300)


# In[137]:


y_kmeans_new = kmean_new.fit(df_new)


# In[138]:


kmean_new.cluster_centers_ #centroid


# In[139]:


kmean_new.n_iter_


# In[140]:


sms.scatterplot(data=df_new,x='Sepal Length', y='Sepal Width', hue=kmean_new.labels_)


# In[141]:


sms.scatterplot(data=df_new,x='Petal Length', y='Petal Width', hue=kmean_new.labels_)


# In[142]:


kmeans_kwargs_new = {
    "init": 'random',
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42
}

sse_new=[]

for k in range (1,20):
    kmeans_new = KMeans(n_clusters=k, **kmeans_kwargs_new)
    kmeans_new.fit(df_new)
    sse_new.append(kmeans_new.inertia_)
    


# In[143]:


plt.style.use('fivethirtyeight')
plt.plot(range(1,20),sse_new)
plt.xticks(range(1,20))
plt.xlabel('Jumlah Klaster')
plt.ylabel('WCSS')
plt.show


# In[144]:


km_new =KneeLocator(range(1,20),sse,curve="convex", direction = "decreasing")


# In[145]:


km_new.elbow


# In[146]:


sc_new = []

for k in range (2,20):
    kmean1 = KMeans(n_clusters=k, **kmeans_kwargs_new)
    kmean1.fit(df_new)
    
    score_new = silhouette_score(df_new, kmean1.labels_)
    sc_new.append(score_new)


# In[147]:


plt.style.use('fivethirtyeight')
plt.plot(range(2,20),sc_new)
plt.xticks(range(2,20))
plt.xlabel('Jumlah Klaster')
plt.ylabel('SC')
plt.show


# In[ ]:




