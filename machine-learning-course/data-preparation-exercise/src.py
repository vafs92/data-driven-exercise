#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("mesin_data_sks_ipk.csv")
df.head()


# In[3]:


df.info()


# In[4]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[5]:


df['Nama']=le.fit_transform(df['Nama'])


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


y=df['ANGKATAN']


# In[9]:


df.drop('ANGKATAN', axis=1, inplace=True)


# In[10]:


df.info()


# In[11]:


updated_df = df.dropna(axis=1)


# In[12]:


updated_df.info()


# In[13]:


# pembagian data training testing
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(updated_df,y,test_size=0.3)
#membangun model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(metrics.accuracy_score(pred,y_test))


# In[14]:


newdf=pd.read_csv("mesin_data_sks_ipk.csv")
newdf.head()


# In[15]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
newdf['Nama']=le.fit_transform(newdf['Nama'])


# In[16]:


newdf.info()


# In[17]:


updated_df=newdf.dropna(axis=0)


# In[18]:


updated_df.info()


# In[19]:


y1=updated_df['ANGKATAN']
updated_df.drop('ANGKATAN',axis=1,inplace=True)


# In[20]:


updated_df.info()


# In[21]:


# pembagian data training testing
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(updated_df,y1,test_size=0.3)
#membangun model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(metrics.accuracy_score(pred,y_test))


# In[22]:


updated_df_fill=df


# In[23]:


updated_df_fill.info()


# In[24]:


updated_df_fill['IPK']=updated_df_fill['IPK'].fillna(updated_df_fill['IPK'].mean())


# In[25]:


updated_df_fill.info()


# In[26]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(updated_df_fill,y,test_size=0.3)
#membangun model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(metrics.accuracy_score(pred,y_test))


# In[27]:


updated_df_si = df
updated_df_si['IPKismissing']=updated_df_si['IPK'].isnull()
from sklearn.impute import SimpleImputer
my_imputer=SimpleImputer(strategy='median')
data_new=my_imputer.fit_transform(updated_df_si)


# In[28]:


updated_df_si.info()


# In[29]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(updated_df_si,y,test_size=0.3)
#membangun model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(metrics.accuracy_score(pred,y_test))


# In[30]:


df.head()


# In[31]:


newdf = df[['IPK', 'SKS']]
newdf


# In[32]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

zscore = StandardScaler()
minmax = MinMaxScaler()

scaler = zscore.fit(newdf)
scaler_m = minmax.fit(newdf)


# In[33]:


df_z = scaler.transform(newdf)
df_z = pd.DataFrame(df_z)
df_z


# In[34]:


df_m = scaler_m.transform(newdf)
df_m = pd.DataFrame(df_m)
df_m


# In[35]:


new_normal = df.copy()
new_normal['Z-SKS'] = df_z[0]
new_normal['Z-IPK'] = df_z[1]
new_normal['MinMax-SKS'] = df_m[0]
new_normal['MinMax-IPK'] = df_m[1]
new_normal


# In[36]:


new_normal_z = new_normal.copy()
new_normal_z = new_normal_z.drop(columns=['SKS', 'IPK', 'IPKismissing', 'MinMax-SKS', 'MinMax-IPK'], axis=1)
new_normal_z


# In[37]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(new_normal_z,y,test_size=0.3)
#membangun model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(metrics.accuracy_score(pred,y_test))


# In[ ]:




