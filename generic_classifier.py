# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 07:52:13 2018

@author: willsm
"""
# pandas is used for data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#read data from cfg

df = pd.read_csv('cfg.csv', index_col = 0, header=None)
path = df.loc['path', 1]

header = df.loc['header', 1:].dropna(axis=0)
# read feature_cols from cfg file
feature_cols = df.loc['feature_cols'].dropna(axis=0)
if df.loc['header',1] == 'None':
    features = pd.read_csv(path, delimiter = ',', na_values = ['?']) # works with t4
else:
   features = pd.read_csv(path, delimiter = ',', na_values = ['?'], names = header) 

# Drop any NaN values 
columns_nan = features[feature_cols].columns[features[feature_cols].isnull().any()]
if columns_nan.size > 0:
    features = features.dropna()
    
#Labels are the values we want to predict
label = df.loc['label', 1]

labels = np.array(features[label])

features = features[feature_cols]

#convert to numpy array
#features = np.array(features)
##print (features)

##Training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#print ('Training Features Shape: ', train_features.shape)
#print('Training Labels Shape: ', train_labels.shape)
#print('Testing Features Shape: ', test_features.shape)
#print('Testing Labels Shape: ', test_labels.shape)
print ('\nUsing Random Forest Classifier\n')
print ('Test Labels')
print(test_labels)
#Create a random forest classfier
clf = RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 1, n_jobs = 1, n_estimators = 10, random_state = 42)
#Train the data using the training sets
clf.fit(train_features, train_labels)
label_predict = clf.predict(test_features)
print('Label Predict')
print (label_predict)

#predict probability
#print(clf.predict_proba(test_features))

#Model accuracy, how often is the classifer correct?
print("Accuracy:", metrics.accuracy_score(test_labels, label_predict))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, label_predict))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, label_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, label_predict)))
#
from sklearn.metrics import classification_report, confusion_matrix#, accuracy_score
print(confusion_matrix(test_labels, label_predict))
print(classification_report(test_labels, label_predict))

#Feature importance
feature_imp = pd.Series(clf.feature_importances_,index=train_features.columns).sort_values(ascending=False)
#print(feature_imp)

#Visualize feature importance
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
#Add labels to graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualiziing Important Features")
#plt.legend()
plt.show()

print('\nUsing Logistic Regression\n')
#
from sklearn.linear_model import LogisticRegression
logregressor = LogisticRegression()
logregressor.fit(train_features, train_labels)
label_predict = logregressor.predict(test_features)

print ('Test Labels')
print(test_labels)

print('Label Predict')
print (label_predict)

print("Accuracy:", metrics.accuracy_score(test_labels, label_predict))

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, label_predict))
#print('Mean Squared Error:', metrics.mean_squared_error(test_labels, label_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, label_predict)))

from sklearn.metrics import classification_report, confusion_matrix#, accuracy_score
print(confusion_matrix(test_labels, label_predict))
print(classification_report(test_labels, label_predict))


