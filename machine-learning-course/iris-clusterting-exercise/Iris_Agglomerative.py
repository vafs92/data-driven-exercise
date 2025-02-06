#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# In[30]:


df=pd.read_csv("iris.csv", sep=",")
df.head()


# In[31]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Label']=le.fit_transform(df['Label'])


# In[32]:


linkage_data = linkage(df, method='single', metric='euclidean')
dendrogram(linkage_data)

plt.title("Single Linkage")
#batas threshold misalkan = 1
max_d=1.5
plt.axhline(y=max_d, c='k')
plt.show()


# In[33]:


linkage_data1 = linkage(df, method='complete', metric='euclidean')
plt.title("Complete Linkage")
dendrogram(linkage_data1)
plt.axhline(y=max_d, c='k')
plt.show()


# In[34]:


linkage_data2 = linkage(df, method='average', metric='euclidean')
plt.title("Average Linkage")
dendrogram(linkage_data2)
plt.axhline(y=max_d, c='k')
plt.show()


# In[35]:


linkage_data3 = linkage(df, method='ward', metric='euclidean')
plt.title("Ward Linkage")
dendrogram(linkage_data3)
plt.axhline(y=max_d, c='k')
plt.show()


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram 

df1=pd.read_csv("iris.csv", sep=",")
df1.head()
le=LabelEncoder()
df1['Label']=le.fit_transform(df1['Label'])

data=pd.DataFrame(df1)
print(data)
#data = list(zip(x, y))

#print(data)

#melakukan clustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
labels = hierarchical_cluster.fit_predict(df1)
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


# In[37]:


#TARGET DIHILANGKAN


# In[38]:


df_new=pd.read_csv("iris.csv", sep=",")
df_new.head()


# In[39]:


df_new.drop("Label",axis=1,inplace=True)
df_new.head()


# In[40]:


linkage_data = linkage(df_new, method='single', metric='euclidean')
dendrogram(linkage_data)

plt.title("Single Linkage")
#batas threshold misalkan = 1
max_d=1.5
plt.axhline(y=max_d, c='k')
plt.show()


# In[41]:


linkage_data1 = linkage(df_new, method='complete', metric='euclidean')
plt.title("Complete Linkage")
dendrogram(linkage_data1)
plt.axhline(y=max_d, c='k')
plt.show()


# In[42]:


linkage_data2 = linkage(df_new, method='average', metric='euclidean')
plt.title("Average Linkage")
dendrogram(linkage_data2)
plt.axhline(y=max_d, c='k')
plt.show()


# In[43]:


linkage_data3 = linkage(df_new, method='ward', metric='euclidean')
plt.title("Ward Linkage")
dendrogram(linkage_data3)
plt.axhline(y=max_d, c='k')
plt.show()


# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram 

df1_new=pd.read_csv("iris.csv", sep=",")
df1_new.head()
df1_new.drop("Label",axis=1,inplace=True)

data_new=pd.DataFrame(df1_new)
print(data_new)
#data = list(zip(x, y))

#print(data)

#melakukan clustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
labels = hierarchical_cluster.fit_predict(df1_new)
print(labels)

linkage_data = linkage(data_new, method='average', metric='euclidean')
dendrogram(linkage_data)
#Threshold 
max_d = 1.5
plt.axhline(y=max_d, c='k')
plt.show()

plt.scatter(data_new['Petal Length'], data_new['Petal Width'],c=labels)
plt.title("Agglomerative")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

plt.scatter(data_new['Sepal Length'], data_new['Sepal Width'] ,c=labels)
plt.title("Agglomerative")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

