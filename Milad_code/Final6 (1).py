
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from keras.models import Sequential
# from keras.layers import Dense


# In[2]:


X_train = pd.read_csv("../datasets/complete_data/Train_DB.csv")
X_test = pd.read_csv("../datasets/complete_data/Test_DB.csv")

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


# In[3]:


X_train.head()


# In[4]:


X_test.head()


# In[5]:


print(X_train.shape)
print(X_test.shape)


# In[6]:


X_train.iloc[42207]


# In[7]:


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


# In[11]:


X_train = df [:42208]
X_test = df [42208:]


# # In[ ]:



# In[ ]:


# RANDOM SEARCH FOR 20 COMBINATIONS OF PARAMETERS
# rand_list = {"C": stats.uniform(2, 10),
#              "gamma": stats.uniform(0.1, 1)}
              
# rand_search = RandomizedSearchCV(model_svm, param_distributions = rand_list, n_iter = 5, n_jobs = 4, cv = 3, random_state = 2017, scoring = auc) 
# rand_search.fit(X_train,y_train) 
# rand_search.cv_results


# In[ ]:

print("plotting")

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


all_y=y_train.append(y_test)
y = label_binarize(all_y, classes=[0, 1, 2])
y_train = y[:42208,:]
y_test = y[42208:,:]
# print(y_train1.shape)
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_pred = classifier.fit(X_train, y_train).decision_function(X_test)
n_classes = 3
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.save("./out.fig")
# plt.show()


# In[ ]:


# import itertools
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting normalize=True.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


# class_names=y_train
# class_names=np.unique(class_names) 
    
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred_svm)
# np.set_printoptions(precision=2)
# # Plot normalized confusion matrix
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()

