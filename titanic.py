#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('/home/shaggy/titanic/train.csv')
test = pd.read_csv('/home/shaggy/titanic/test.csv')

PassengerId = test['PassengerId']

train.head()


# In[3]:


for dataset in [train, test]:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    dataset['HasCabin'] = dataset['Cabin'].apply(lambda x : 1 if type(x) == float else 0)

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    dataset['IsAlone'] = 0
    dataset.loc[train['FamilySize'] == 0, 'IsAlone'] = 1

    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    dataset['CategoryAge'] = pd.cut(dataset['Age'], 4)

    dataset.loc[dataset['Age'] < 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] >= 20) & (train['Age'] < 40), 'Age'] = 1
    dataset.loc[(dataset['Age'] >= 40) & (train['Age'] < 60), 'Age'] = 2
    dataset.loc[(dataset['Age'] >= 60), 'Age'] = 3

    dataset['CategoryFare'] = pd.cut(train['Fare'], 4)

    dataset.loc[train['Fare'] < 128.082, 'Fare'] = 0
    dataset.loc[(train['Fare'] >= 128.082) & (dataset['Fare'] < (128.082 * 2)), 'Age'] = 1
    dataset.loc[(train['Fare'] >= (128.082 * 2)) & (dataset['Fare'] < (128.082 * 3)), 'Age'] = 2
    dataset.loc[(train['Fare'] >= (128.082 * 3)), 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

drop = ['Name', 'PassengerId', 'Ticket', 'FamilySize', 'SibSp', 'Parch', 'Cabin', 'CategoryFare', 'CategoryAge']    
train = train.drop(drop, axis=1)
test = test.drop(drop, axis=1)

test.head(10)


# In[4]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[5]:


y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values 


# In[6]:


gbm = xgb.XGBClassifier().fit(x_train, y_train)
xgb_predictions = gbm.predict(x_test)
xgb_predictions


# In[7]:


StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': xgb_predictions })

# Generate Submission File 
StackingSubmission.to_csv("XGB.csv", index=False)
StackingSubmission.head()

