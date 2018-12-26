
# %%-----------------------------------------------------------------------
# Import packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)
os.chdir("/Users/Kim/Documents/GWU Class/Machine Learning I/Final-Project-Group5-master/all")

# %%-----------------------------------------------------------------------
# Import the data
data=pd.read_csv("application_train.csv")
print(data.head())
print(data.isnull().sum())
print('\n')
# Drop rows with missing values
data=data.dropna()
data.info()
print('\n')
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])
print('\n')
print(data.describe(include='all'))
print('\n')
print(list(data))
print('\n')
print(data.dtypes)
print('\n')
print(data["TARGET"].value_counts())
print('\n')
#
# # %%-----------------------------------------------------------------------
# # Plot the data
# # Plot group means for numerical variables
# def num_var_plot(var):
#     graphdata=data.groupby('TARGET')[var].mean().reset_index()
#     target = {0:"No default", 1:"Default"}
#     graphdata['TARGET'].replace(target, inplace=True)
#     print(graphdata)
#     x=graphdata['TARGET']
#     y=graphdata[var]
#     plt.bar(x, y, align='center',width=0.6,color='navy')
#     plt.xticks(x)
#     plt.ylabel(var)
#     plt.title(var+' BY TARGET')
#     plt.rc('font',family='Times New Roman')
#     plt.show()
#
# num_var_plot('AMT_INCOME_TOTAL')
# num_var_plot('CNT_CHILDREN')
# num_var_plot('DAYS_BIRTH')
#
# # Plot for categorical variables
# def cat_var_plot(var):
#     data['freq']=1
#     graphdata=data.groupby(['TARGET',var])['freq'].count().reset_index()
#     target = {0:"No default", 1:"Default"}
#     graphdata['TARGET'].replace(target, inplace=True)
#     graphdata=graphdata.pivot(index='TARGET',columns=var,values='freq')
#     print(graphdata)
#     graphdata.plot.bar(stacked=True)
#     plt.xticks(rotation='horizontal')
#     plt.xlabel('Loan')
#     plt.ylabel('Frequency')
#     plt.rc('font', family='Times New Roman')
#     plt.show()
#
# cat_var_plot('NAME_CONTRACT_TYPE')
# cat_var_plot('CODE_GENDER')
# cat_var_plot('FLAG_OWN_REALTY')

# %%-----------------------------------------------------------------------
# Data pre-processing
# Normalize the numerical variables
# nor_columns=['AMT_INCOME_TOTAL','DAYS_BIRTH','CNT_CHILDREN','CNT_FAM_MEMBERS']
# data['DAYS_BIRTH']=abs(data['DAYS_BIRTH'])
# for var in nor_columns:
#     data[var] = (data[var] - data[var].min()) / (data[var].max() - data[var].min())
#     print(data[var].describe())
#     print('\n')

# Encode all the categorical variables
obj_columns = data.select_dtypes(include=['object']).columns
print(obj_columns)
data[obj_columns] = data[obj_columns].astype('category')
data[obj_columns] = data[obj_columns].apply(lambda x: x.cat.codes)

# # Select variables to be included in the analysis
# data = data[['TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN',
#              'AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE','DAYS_BIRTH','REGION_RATING_CLIENT',
#              'AMT_REQ_CREDIT_BUREAU_DAY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','OCCUPATION_TYPE',
#              'REG_REGION_NOT_LIVE_REGION']]
data_cash=data[data.NAME_CONTRACT_TYPE==0]
data_rev=data[data.NAME_CONTRACT_TYPE==1]
print(data.describe(include='all'))

# %%-----------------------------------------------------------------------
# Re-sample the data to reduce the over-represented class
print(data["TARGET"].value_counts())
print('\n')
data1=data.loc[data['TARGET']==1]
data0=data.loc[data['TARGET']==0].sample(n=data1.shape[0])
print(data1.shape)
print(data0.shape)
data=pd.concat([data0,data1])
print(data.shape)
# split the dataset
# separate the target variable
x = data.values[:, 2:]
y = data.values[:, 1]

# encloding the class with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(y)

# split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
#
# # %%-----------------------------------------------------------------------
# # Perform training: SVC
# # creating the classifier object
# clf = SVC(kernel="linear")
# # performing training
# clf.fit(x_train, y_train)
# # make predictions
# # predicton on test
# y_pred = clf.predict(x_test)
#
# # calculate metrics
# print("\n")
# print("Classification Report")
# print(classification_report(y_test,y_pred))
# print("\n")
# print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
# print("\n")
#
# # confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_names = data['TARGET'].unique()
#
# df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
# plt.figure(figsize=(5,5))
# hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
# hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
# hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
# plt.ylabel('True label',fontsize=20)
# plt.xlabel('Predicted label',fontsize=20)
# # Show heat map
# plt.tight_layout()
# plt.show()
#
# # Plot ROC Area Under Curve
# y_pred_proba = clf.decision_function(x_test)
# fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
# auc = roc_auc_score(y_test, y_pred_proba)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic Curve')
# plt.legend(loc="lower right")
# plt.show()

# %%-----------------------------------------------------------------------
# Perform training:
#Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

rf = RandomForestClassifier(n_estimators=100)

# perform training
rf.fit(x_train, y_train)


#plot feature importances
# get feature importances
importances = rf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.iloc[:, 2:].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(121, 9), rot=90, fontsize=6)

# show the plot
plt.tight_layout()
plt.show()

# predicton on test using all features

y_pred = rf.predict(x_test)
y_pred_score = rf.predict_proba(x_test)


# calculate metrics all features

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)



#perform training with random forest with k columns
# specify random forest classifier
rf_k_features = RandomForestClassifier(n_estimators=100)


# select the training dataset on k-features
newX_train = x_train[:, rf.feature_importances_.argsort()[::-1][:30]]

# select the testing dataset on k-features
newX_test = x_test[:, rf.feature_importances_.argsort()[::-1][:30]]
# train the model
rf_k_features.fit(newX_train, y_train)
# prediction on test using k features
y_pred_k_features = rf_k_features.predict(newX_test)
y_pred_k_features_score = rf_k_features.predict_proba(newX_test)

print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)

#%%-----------------------------------------------------------------------


# %%-----------------------------------------------------------------------
# confusion matrix for all features
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['TARGET'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
##########################################
# confusion matrix for k features
conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = data['TARGET'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()