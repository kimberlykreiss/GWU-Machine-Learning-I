# DATS 6202 Final Project
# August 2018
# Individual Code: Zhoudan Xie

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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)

# %%-----------------------------------------------------------------------
# Import the data
data=pd.read_csv("application_train.csv")
print(data.head())
print(data.isnull().sum())
print('\n')
# Drop rows with missing values
data=data.dropna()
# Look at the structure of the data
data.info()
print('\n')
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])
print('\n')
print(data.dtypes)
print('\n')
print(data["TARGET"].value_counts())
print('\n')

# %%-----------------------------------------------------------------------
# Plot the data
# Plot group means for numerical variables
def num_var_plot(var):
    graphdata=data.groupby('TARGET')[var].mean().reset_index()
    target = {0:"No default", 1:"Default"}
    graphdata['TARGET'].replace(target, inplace=True)
    print(graphdata)
    x=graphdata['TARGET']
    y=graphdata[var]
    plt.bar(x, y, align='center',width=0.6,color='navy')
    plt.xticks(x)
    plt.ylabel(var)
    plt.title(var+' BY TARGET')
    plt.rc('font',family='Times New Roman')
    plt.show()

num_var_plot('AMT_INCOME_TOTAL')
num_var_plot('CNT_CHILDREN')
num_var_plot('DAYS_BIRTH')

# Plot for categorical variables
def cat_var_plot(var):
    data['freq']=1
    graphdata=data.groupby(['TARGET',var])['freq'].count().reset_index()
    target = {0:"No default", 1:"Default"}
    graphdata['TARGET'].replace(target, inplace=True)
    graphdata=graphdata.pivot(index='TARGET',columns=var,values='freq')
    print(graphdata)
    graphdata.plot.bar(stacked=True)
    plt.xticks(rotation='horizontal')
    plt.xlabel('Loan')
    plt.ylabel('Frequency')
    plt.rc('font', family='Times New Roman')
    plt.show()

cat_var_plot('NAME_CONTRACT_TYPE')
cat_var_plot('CODE_GENDER')
cat_var_plot('FLAG_OWN_REALTY')


# %%-----------------------------------------------------------------------
# Support Vector Machine
# Data pre-processing
data_svm=data

# Select variables to be included in the analysis
data_svm = data_svm[['TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN',
             'AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE','DAYS_BIRTH','REGION_RATING_CLIENT',
             'AMT_REQ_CREDIT_BUREAU_DAY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','OCCUPATION_TYPE',
             'REG_REGION_NOT_LIVE_REGION']]
print("Dataset No. of Rows: ", data_svm.shape[0])
print("Dataset No. of Columns: ", data_svm.shape[1])
print('\n')
print(data_svm.dtypes)
print('\n')

# Normalize the numerical variables
nor_columns=['AMT_INCOME_TOTAL','DAYS_BIRTH','CNT_CHILDREN','AMT_REQ_CREDIT_BUREAU_DAY']
data_svm['DAYS_BIRTH']=abs(data_svm['DAYS_BIRTH'])
for var in nor_columns:
    data_svm[var] = (data_svm[var] - data_svm[var].min()) / (data_svm[var].max() - data_svm[var].min())
    print(data_svm[var].describe())
    print('\n')

# Encode all the categorical variables
obj_columns = data_svm.select_dtypes(include=['object']).columns
print(obj_columns)
data_svm[obj_columns] = data_svm[obj_columns].astype('category')
data_svm[obj_columns] = data_svm[obj_columns].apply(lambda x: x.cat.codes)

# Re-sample the data to reduce the over-represented class
print(data_svm["TARGET"].value_counts())
print('\n')
data1=data_svm.loc[data['TARGET']==1]
data0=data_svm.loc[data['TARGET']==0].sample(n=data1.shape[0])
print(data1.shape)
print(data0.shape)
data_svm=pd.concat([data0,data1])
print(data_svm.shape)

# %%-----------------------------------------------------------------------
# Split the dataset
# Separate the target variable
x = data_svm.values[:, 1:]
y = data_svm.values[:, 0]

# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# %%-----------------------------------------------------------------------
# Perform SVM training
# Create the classifier object
def svm(kernel):
    clf = SVC(kernel=kernel)
    # perform training
    clf.fit(x_train, y_train)
    # make predictions on test
    y_pred = clf.predict(x_test)

    # calculate metrics
    print("\n")
    print("Classification Report (Kernel="+kernel+")")
    print(classification_report(y_test,y_pred))
    print("\n")
    print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
    print("\n")

    # confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = data_svm['TARGET'].unique()

    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
    plt.figure(figsize=(5,5))
    hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label',fontsize=14)
    plt.xlabel('Predicted label',fontsize=14)
    plt.title('Confusion Matrix (Kernel='+kernel+')',fontsize=16)
    plt.rc('font', family='Times New Roman')
    # Show heat map
    plt.tight_layout()
    plt.show()

    # plot ROC Area Under Curve
    y_pred_proba = clf.decision_function(x_test)
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('Receiver Operating Characteristic Curve (Kernel='+kernel+')',fontsize=16)
    plt.legend(loc="lower right")
    plt.rc('font', family='Times New Roman')
    plt.show()

svm('linear')
svm('rbf')
svm('poly')
svm('sigmoid')