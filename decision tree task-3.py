#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('C:\\Users\\v.omsai\\Downloads\\bank.csv',sep = ';')
df


# In[3]:


df['job'].value_counts()


# In[4]:


df['marital'].value_counts()


# In[5]:


df['education'].value_counts()


# In[6]:


df['default'].value_counts()


# In[7]:


df['balance'].value_counts()


# In[8]:


df['housing'].value_counts()


# In[9]:


df['loan'].value_counts()


# In[10]:


df['contact'].value_counts()


# In[11]:


df['month'].value_counts()


# In[12]:


df['poutcome'].value_counts()


# In[13]:


df['y'].value_counts()


# In[14]:


df.describe()


# In[15]:


category = ['job','marital','education','default','housing','loan','contact','month','poutcome']


# In[16]:


df[category].describe()


# In[17]:


df.isnull().sum()


# In[18]:


sns.pairplot(df)


# In[19]:


sns.heatmap(df.corr(),annot=True)


# In[20]:


plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(1,6,1)
sns.histplot(df['age'],kde=True)

plt.subplot(1,6,2)
sns.histplot(df['balance'],kde=True)

plt.subplot(1,6,3)
sns.histplot(df['day'],kde=True)

plt.subplot(1,6,4)
sns.histplot(df['duration'],kde=True)

plt.subplot(1,6,5)
sns.histplot(df['campaign'],kde=True)

plt.subplot(1,6,6)
sns.histplot(df['previous'])
plt.show()


# In[21]:


df['balance'].skew()


# In[22]:


df['duration'].skew()


# In[23]:


df['campaign'].skew()


# In[24]:


df['previous'].skew()


# In[25]:


from scipy.stats import boxcox
df['balance'],a = boxcox(abs(df['balance']+0.000001))
df['duration'],b = boxcox(abs(df['duration']+0.0000001))
df['campaign'],c = boxcox(abs(df['campaign']+0.00001))
df['previous'],d = boxcox(abs(df['previous']+0.000001))


# In[30]:


df = df.drop(columns=['month'])


# In[31]:


X = pd.get_dummies(df.drop('y',axis=1),drop_first=True)
y = df['y']


# In[35]:


train = []
test = []
cv = []
for i in range(1,101):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=i)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    ypred_train = model.predict(X_train)
    ypred_test = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    train.append(accuracy_score(ypred_train,y_train))
    test.append(accuracy_score(ypred_test,y_test))
    
    from sklearn.model_selection import cross_val_score
    cv.append(cross_val_score(model,X,y,cv=5,scoring='accuracy').mean())
em = pd.DataFrame({'Train':train,'Test':test,'CV':cv})
gm = em[(abs(em['Train']-em['Test'])<=0.05) & (abs(em['Test']-em['CV'])<=0.05)]
rs = gm[gm['Train']==gm['Train'].max()].index.to_list()
print(rs)


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=24)


# In[48]:


from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier()
param_grid = {'criterion':['gini','entropy'],
              'max_depth':list(range(1,10))}
from sklearn.model_selection import GridSearchCV
tree_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
tree_grid.fit(X_train,y_train)

model_tree = tree_grid.best_estimator_

tree_feat = model_tree.feature_importances_

index = [i for i,x in enumerate(tree_feat) if x>0]

X_train_tree = X_train.iloc[:,index]
X_test_tree = X_test.iloc[:,index]

model_tree.fit(X_train_tree,y_train)

ypred_train = model_tree.predict(X_train_tree)
ypred_test = model_tree.predict(X_test_tree)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model_tree,X_train_tree,y_train,cv=5,scoring='accuracy').mean())


# In[ ]:




