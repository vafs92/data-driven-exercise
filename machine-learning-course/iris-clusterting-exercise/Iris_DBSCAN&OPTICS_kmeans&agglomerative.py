#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics


# In[157]:


X = pd.read_csv("iris.csv", sep=",")
# X.info()


# In[158]:


X = X.drop("Label", axis=1)
X


# In[159]:


# DBSCAN

db = DBSCAN(eps=1,min_samples=25).fit(X)
labels = db.labels_
print(labels)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# In[160]:


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


# In[161]:


print ('Estimated number of clus: %d' % n_clusters_)
print('Estimated number of noise points: %d'%n_noise_)
print('Sihouette Coefficient: %0.3f'%metrics.silhouette_score(X, labels))


# In[162]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
         for each in np.linspace(0,1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k ==-1:
        col=[0,0,0,1]
        
    class_member_mask = (labels == k)
    
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1],"o",markerfacecolor=tuple(col),
            markeredgecolor='k', markersize = 14)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1],"o",markerfacecolor=tuple(col),
            markeredgecolor='k', markersize = 6)
    
plt.title ('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[163]:


#visualisasi

colours = {}
colours[0]='r'
colours[1]='g'
colours[2]='b'
colours[-1]='k'

cvec = [colours[label]for label in labels]

r = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'r');
g = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'g');
b = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'b');
k = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'k');

plt.figure(figsize =(9,9))
plt.scatter(X['Sepal Length'], X['Sepal Width'], c = cvec)

plt.legend((r,g,b,k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

plt.show()


# In[164]:


#visualisasi

colours = {}
colours[0]='r'
colours[1]='g'
colours[2]='b'
colours[-1]='k'

cvec = [colours[label]for label in labels]

r = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'r');
g = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'g');
b = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'b');
k = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'k');

plt.figure(figsize =(9,9))
plt.scatter(X['Petal Length'], X['Petal Width'], c = cvec)

plt.legend((r,g,b,k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

plt.show()


# In[165]:


#OPTICS

from sklearn.cluster import OPTICS
db = OPTICS(min_samples=25).fit(X)
labels = db.labels_
print(labels)


# In[166]:


#visualisasi

colours = {}
colours[0]='r'
colours[1]='g'
colours[2]='b'
colours[-1]='k'

cvec = [colours[label]for label in labels]

r = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'r');
g = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'g');
b = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'b');
k = plt.scatter(X['Sepal Length'], X['Sepal Width'], color = 'k');

plt.figure(figsize =(9,9))
plt.scatter(X['Sepal Length'], X['Sepal Width'], c = cvec)

plt.legend((r,g,b,k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

plt.show()


# In[167]:


#visualisasi

colours = {}
colours[0]='r'
colours[1]='g'
colours[2]='b'
colours[-1]='k'

cvec = [colours[label]for label in labels]

r = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'r');
g = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'g');
b = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'b');
k = plt.scatter(X['Petal Length'], X['Petal Width'], color = 'k');

plt.figure(figsize =(9,9))
plt.scatter(X['Petal Length'], X['Petal Width'], c = cvec)

plt.legend((r,g,b,k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

plt.show()


# In[168]:


#KMEANS

from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=2, random_state=0, algorithm='auto', init='k-means++', max_iter=300)
y_kmeans = kmean.fit(X)
kmean.cluster_centers_ #centroid


# In[169]:


kmean.labels_


# In[170]:


import seaborn as sms
sms.scatterplot(data=X,x='Sepal Length', y='Sepal Width', hue=kmean.labels_)


# In[171]:


sms.scatterplot(data=X,x='Petal Length', y='Petal Width', hue=kmean.labels_)


# In[172]:


kmeans_kwargs = {
    "init": 'random',
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42
}

sse=[]

for k in range (1,20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    


# In[173]:


#metode elbow

plt.style.use('fivethirtyeight')
plt.plot(range(1,20),sse)
plt.xticks(range(1,20))
plt.xlabel('Jumlah Klaster')
plt.ylabel('WCSS')
plt.show


# In[174]:


from kneed import KneeLocator
km =KneeLocator(range(1,20),sse,curve="convex", direction = "decreasing")


# In[175]:


km.elbow


# In[176]:


from sklearn.metrics import silhouette_score
#silhouette_cofficient
sc = []

for k in range (2,20):
    kmean = KMeans(n_clusters=k, **kmeans_kwargs)
    kmean.fit(X)
    
    score = silhouette_score(X, kmean.labels_)
    sc.append(score)


# In[177]:


plt.style.use('fivethirtyeight')
plt.plot(range(2,20),sc)
plt.xticks(range(2,20))
plt.xlabel('Jumlah Klaster')
plt.ylabel('SC')
plt.show


# In[178]:


#AGGLOMERATIVE

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram 


data=pd.DataFrame(X)
print(data)

#melakukan clustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
labels = hierarchical_cluster.fit_predict(X)
print(labels)

linkage_data = linkage(data, method='average', metric='euclidean')
dendrogram(linkage_data)
#Threshold 
max_d = 1.5
plt.axhline(y=max_d, c='k')
plt.show()

plt.scatter(data['Petal Length'], data['Petal Width'],c=labels)
plt.title("Agglomerative")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

plt.scatter(data['Sepal Length'], data['Sepal Width'] ,c=labels)
plt.title("Agglomerative")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




