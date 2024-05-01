#!/usr/bin/env python
# coding: utf-8

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[13]:


import warnings

warnings.filterwarnings('ignore')


# In[15]:


data = '/kaggle/input/facebook-live-sellers-in-thailand-uci-ml-repo/Live.csv'

df = pd.read_csv('Live.csv')
print(df.to_string())


# In[17]:


df.shape


# In[19]:


df.head()


# In[20]:


df.info()


# In[21]:


df.isnull().sum()


# In[22]:


df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)


# In[23]:


df.info()


# In[24]:


df.describe()


# In[25]:


# view the labels in the variable

df['status_id'].unique()


# In[26]:


# view how many different types of variables are there

len(df['status_id'].unique())


# In[27]:


# view the labels in the variable

df['status_published'].unique()


# In[28]:


# view how many different types of variables are there

len(df['status_published'].unique())


# In[29]:


# view the labels in the variable

df['status_type'].unique()


# In[30]:


# view how many different types of variables are there

len(df['status_type'].unique())


# In[31]:


df.drop(['status_id', 'status_published'], axis=1, inplace=True)


# In[32]:


df.info()


# In[33]:


df.head()


# In[34]:


X = df

y = df['status_type']


# In[35]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['status_type'] = le.fit_transform(X['status_type'])

y = le.transform(y)


# In[36]:


X.info()


# In[37]:


X.head()


# In[39]:


# feature scaling
cols = X.columns


# In[40]:


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)


# In[42]:


X = pd.DataFrame(X, columns=[cols])


# In[43]:


X.head()


# In[44]:


# k-means model with two model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(X)


# In[45]:


# k-means model parameter study
kmeans.cluster_centers_


# In[46]:


kmeans.inertia_


# In[48]:


# check quality of weak classification by the model
labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[49]:


print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[50]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[51]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=0)

kmeans.fit(X)

labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[52]:


# k-means model with different clusters
kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[53]:


# with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# We have achieved a relatively high accuracy of 62% with k=4.
