#!/usr/bin/env python
# coding: utf-8

# # MH6804: Python For Data Analysis

# # Group Project: Credit Card Fraud

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os


# In[2]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the data

# In[3]:


df = pd.read_csv("../creditcard.csv")
df


# # EDA: Exploration on the data

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


# Finding the missing value
df.isnull().sum()


# In[9]:


#cross check if there is any nna value?
df.isna().any()


# # Finding the data distribution 

# In[10]:


#finding the count of each value for that parameter. meaning we are finding how many fraud and real
df['Class'].value_counts()


# #### Class -> Real = 0; Fraud = 1

# In[11]:


df_real = df[df['Class'] == 0]
df_fraud = df[df['Class'] == 1]

type(df_real)

type(df_fraud)


# In[12]:


df_real.sum()


# In[13]:


df_fraud.sum()


# In[14]:


ax = df_real.plot.box()


# In[15]:


ax = df_fraud.plot.box()


# In[16]:


#finding the count of each value for that parameter. meaning we are finding how many fraud and real
df['Class'].value_counts()


# In[17]:


# Visualizing the distribution of the 'Class' column (Target variable)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Class')
plt.title('Distribution of Fraudulent vs. Non-Fraudulent Transactions')
plt.show()


# ### Fraud is too few

# In[18]:


# Visualizing the distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()


# In[19]:


# Correlation matrix to understand relationships between features
plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False, fmt=".1f", linewidths=0.1)
plt.title('Correlation Matrix')
plt.show()


# In[20]:


df.corrwith(df.Class).plot.bar(
        figsize = (20, 10), title = "Correlation with class", fontsize = 15,
        rot = 45, grid = True)


# ### We can see that the columns that relatively correlates with the class are:
# 
# ### Positive Correlation
# - V11
# 
# ### Negative Correlation
# - V3
# - V10
# - V12
# - V14
# - V16
# - V17
# 

# # Feature Scaling

# #### We normalize the "Amount" column so that its large numbers don't overshadow or unfairly impact the other pieces of information when we run our analysis.

# In[21]:


# Normalizing the amount data
from sklearn.preprocessing import StandardScaler
df['normalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df.head()


# In[22]:


df["normalizedAmount"].max()


# In[23]:


df["normalizedAmount"].min()


# #### Dropping the "Time" column because the exact time of a transaction may not provide meaningful information for detecting fraud, and leaving it in might introduce noise or distractions to the analysis.

# In[24]:


#dropping the time
df_drop_time = df.drop(['Time'],axis=1)
df_drop_time.head()


# In[25]:


df_drop_time_real = df_drop_time[df_drop_time['Class'] == 0]
df_drop_time_fraud = df_drop_time[df_drop_time['Class'] == 1]


# In[26]:


ax = df_drop_time_real.plot.box()


# In[27]:


ax = df_drop_time_fraud.plot.box()


# ### showing the pairplot for relatively high correlation data

# In[28]:


# Calculating columns with high correlation to 'Class'
correlation_with_class = df_drop_time.corrwith(df_drop_time.Class)
high_correlation_columns = correlation_with_class[correlation_with_class.abs() > 0.2].index.drop('Class').tolist()

# Plotting the pairplot
sns.set()
sns.pairplot(df_drop_time[high_correlation_columns + ['Class']], hue='Class', height=2.5)
plt.show()


# In[29]:


# Calculating columns with high correlation to 'Class'
correlation_with_class = df_drop_time.corrwith(df_drop_time.Class)
high_correlation_columns = correlation_with_class[correlation_with_class.abs() > 0.3].index.tolist()

# Plotting the pairplot for all these columns against each other, including 'Class'
sns.set()
sns.pairplot(df_drop_time[high_correlation_columns], hue='Class', height=2.5)
plt.show()


# # Spliting the Data

# In[30]:


X = df_drop_time.iloc[:, df_drop_time.columns != 'Class'].values
y = df_drop_time.iloc[:, df_drop_time.columns == 'Class'].values.ravel()  # ravel might be necessary to create a 1D array


# # Exploring the Model

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# ### *Applying SMOTE to handle the class imbalance*

# In[32]:


from imblearn.over_sampling import SMOTE
from collections import Counter
# Summarize class distribution
print("Before SMOTE: ", Counter(y_train))

# Apply SMOTE
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Summarize the new class distribution
print("After SMOTE: ", Counter(y_train_res))

# model.fit(X_train_res, y_train_res)


# # Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

dt_classifier = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=2)
dt_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = dt_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Extracting TP, FP, TN, and FN from the confusion matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# Computing TPR, FPR, TNR, and FNR
TPR = TP / (TP + FN)  # Same as Recall
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)
FNR = FN / (TP + FN)

results = pd.DataFrame([['Decision tree', acc, prec, rec, f1, TPR, FPR, TNR, FNR]],
               columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TPR', 'FPR', 'TNR', 'FNR'])


# In[34]:


results


# ### Random Forest

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

rf_classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')
rf_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = rf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)  # This is the same as TPR
f1 = f1_score(y_test, y_pred)

# Extracting TP, FP, TN, and FN from the confusion matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# Computing TPR, FPR, TNR, and FNR
TPR = TP / (TP + FN)  
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)
FNR = FN / (TP + FN)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1, TPR, FPR, TNR, FNR]],
               columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TPR', 'FPR', 'TNR', 'FNR'])

results = pd.concat([results, model_results]).reset_index(drop=True)


# In[36]:


results


# ![Screenshot%202023-10-31%20at%208.47.02%20PM.png](attachment:Screenshot%202023-10-31%20at%208.47.02%20PM.png)

# ![Screenshot%202023-10-31%20at%208.47.37%20PM.png](attachment:Screenshot%202023-10-31%20at%208.47.37%20PM.png)

# ## **Models with SMOTE**

# ### Decision Tree with SMOTE

# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

dts_classifier = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=2)
dts_classifier.fit(X_train_res, y_train_res)

# Predicting Test Set
y_pred = dts_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Extracting TP, FP, TN, and FN from the confusion matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# Computing TPR, FPR, TNR, and FNR
TPR = TP / (TP + FN)  # Same as Recall
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)
FNR = FN / (TP + FN)

model_results = pd.DataFrame([['Decision tree SMOTE', acc, prec, rec, f1, TPR, FPR, TNR, FNR]],
               columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TPR', 'FPR', 'TNR', 'FNR'])

results = pd.concat([results, model_results]).reset_index(drop=True)



# In[38]:


results


# ### Random Forest with SMOTE

# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

rfs_classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')
rfs_classifier.fit(X_train_res, y_train_res)

# Predicting Test Set
y_pred = rfs_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)  # This is the same as TPR
f1 = f1_score(y_test, y_pred)

# Extracting TP, FP, TN, and FN from the confusion matrix
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

# Computing TPR, FPR, TNR, and FNR
TPR = TP / (TP + FN)  
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)
FNR = FN / (TP + FN)

model_results = pd.DataFrame([['Random Forest (n=100)SMOTE', acc, prec, rec, f1, TPR, FPR, TNR, FNR]],
               columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TPR', 'FPR', 'TNR', 'FNR'])

results = pd.concat([results, model_results]).reset_index(drop=True)


# In[40]:


results


# In[41]:


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Function to calculate and plot ROC AUC
def plot_roc_auc(classifier, X_test, y_test, model_name):
    # Calculate the probabilities of each class
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)

    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Plot ROC Curve
    plt.plot(fpr, tpr, label=f'{model_name} (area = %0.2f)' % roc_auc)

    return roc_auc

# Plotting both ROC Curves
plt.figure(figsize=(10, 8))

# Plot ROC for Decision Tree & Random Forest without SMOTE
roc_auc_dt = plot_roc_auc(dt_classifier, X_test, y_test, 'Decision Tree')
roc_auc_rf = plot_roc_auc(rf_classifier, X_test, y_test, 'Random Forest')

# Plot ROC for Decision Tree & Random Forest with SMOTE
roc_auc_dt_smote = plot_roc_auc(dts_classifier, X_test, y_test, 'Decision Tree with SMOTE')
roc_auc_rf_smote = plot_roc_auc(rfs_classifier, X_test, y_test, 'Random Forest with SMOTE')

# Plot diagonal
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Append ROC AUC to the results dataframe
results.loc[results['Model'] == 'Decision tree', 'ROC AUC'] = roc_auc_dt
results.loc[results['Model'] == 'Random Forest (n=100)', 'ROC AUC'] = roc_auc_rf
results.loc[results['Model'] == 'Decision tree with SMOTE', 'ROC AUC'] = roc_auc_dt_smote
results.loc[results['Model'] == 'Random Forest (n=100) with SMOTE', 'ROC AUC'] = roc_auc_rf_smote


# ### Note: Please note the result above is after implementing SMOTE without Cross Validation

# # SMOTE with Cross-Validation for Data imbalance

# In[ ]:





# In[ ]:


from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Define the classifier
dt_classifier = DecisionTreeClassifier(random_state=0)
rf_classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')

# Define scoring metrics you want to use
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

# Create a pipeline with SMOTE and Decision Tree
pipeline_dt = imbpipeline([
    ('SMOTE', SMOTE(random_state=0)),
    ('classifier', dt_classifier)
])

# Create a pipeline with SMOTE and Random Forest
pipeline_rf = imbpipeline([
    ('SMOTE', SMOTE(random_state=0)),
    ('classifier', rf_classifier)
])

# Perform cross-validation and store results
results_dt = cross_validate(pipeline_dt, X, y, cv=5, scoring=scoring)
results_rf = cross_validate(pipeline_rf, X, y, cv=5, scoring=scoring)



# Print the results
print("Decision Tree with Cross-Validation and SMOTE:", results_dt)
print("Random Forest with Cross-Validation and SMOTE:", results_rf)


# In[ ]:


type(results_dt)


# In[ ]:


results_dt = pd.DataFrame([results_dt])
results_rf = pd.DataFrame([results_rf])
results_dt['classifier'] = 'Decision Tree'
results_rf['classifier'] = 'Random Forest'


# In[ ]:


results_dt


# In[ ]:


results_rf


# In[ ]:




