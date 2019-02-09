
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


X_train = pd.read_csv("./Train_DB.csv")
X_test = pd.read_csv("./Test_DB.csv")

print(X_test.shape)
print(X_train.shape)

X_train["date"] = pd.to_datetime(X_train["date"] )
X_test["Year_com"] = pd.to_datetime(X_test["Year_com"] )


X_train['Year'] = X_train.date.dt.year
X_train['Month'] = X_train.date.dt.month
X_train['day'] = X_train.date.dt.day



X_test['Year'] = X_test.Year_com.dt.year
X_test['Month'] = X_test.Year_com.dt.month
X_test['day'] = X_test.Year_com.dt.day


X_train= X_train.drop(['date'],axis=1)
X_test = X_test.drop(['Year_com'],axis=1)


y_train =X_train['Winner_num']
y_test =X_test['Winner_num']


X_train= X_train.drop(['Unnamed: 0','home_score','away_score','tournament','winner','city','country','Winner_num'],axis=1)
X_test = X_test.drop(['Unnamed: 0','Score1','Score2','winner','Winner_num'],axis=1)



cols =X_test.columns
X_train.columns = cols
X_train.head()


# In[8]:


df = X_train.append(X_test)
df.head()


# In[9]:


countries = list(set(df["Team1"]).union(set(df["Team2"])))
countries_dict = {}
for i in range(len(countries)):
    countries_dict[countries[i]] = i
    
Team1_encoded = []
Team2_encoded = []

for i in range(len(df)):
    Team1_encoded.append(countries_dict[list(df.Team1)[i]])
    Team2_encoded.append(countries_dict[list(df.Team2)[i]])
    
df["Team1"] = Team1_encoded
df["Team2"] = Team2_encoded


# In[10]:


df.head()


# In[ ]:


X_train = df [:42208]
X_test = df [42208:]


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 


from sklearn import svm
model_svm = svm.SVC(probability = True)
model_svm.fit(X_train,y_train)

y_pred_svm = model_svm.predict(X_test)

print("#################################")
print(accuracy_score(y_test, y_pred_svm))


# In[ ]:


# RANDOM SEARCH FOR 20 COMBINATIONS OF PARAMETERS
rand_list = {"C": stats.uniform(2, 10),
             "gamma": stats.uniform(0.1, 1)}
              
rand_search = RandomizedSearchCV(model_svm, param_distributions = rand_list, n_iter = 10, n_jobs = 4, cv = 3, random_state = 2017, scoring = auc) 
rand_search.fit(X_train,y_train) 
print(rand_search.cv_results)
print(rand_search.best_params_)


