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


df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
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


# In[31]:


#what is the shape of the data?
X.shape


# In[32]:


#what is the shape of the data?
y.shape


# # Exploring the Model

# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# In[34]:


print(f"X_train shape:{X_train.shape}")
print(f"y_train shape:{y_train.shape}")
print(f"X_test shape:{X_test.shape}")
print(f"y_test shape:{y_test.shape}")


# ### *Applying SMOTE to handle the class imbalance*

# In[35]:


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


# In[36]:


# Summarize class distribution
print("Before SMOTE: ", Counter(y_test))

# Generate SMOTE-resampled test data
X_test_res, y_test_res = sm.fit_resample(X_test, y_test)

# Summarize the new class distribution
print("After SMOTE: ", Counter(y_test_res))


# # Decision Tree

# In[39]:


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


# In[40]:


results


# ### Random Forest

# In[41]:


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


# In[42]:


results


# ![Screenshot%202023-10-31%20at%208.47.02%20PM.png](attachment:Screenshot%202023-10-31%20at%208.47.02%20PM.png)

# ![Screenshot%202023-10-31%20at%208.47.37%20PM.png](attachment:Screenshot%202023-10-31%20at%208.47.37%20PM.png)

# ## **Models with SMOTE**

# ### Decision Tree with SMOTE

# In[43]:


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


# In[44]:


results


# ### Random Forest with SMOTE

# In[45]:


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


# In[46]:


results


# In[47]:


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

# In[48]:


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


# In[49]:


type(results_dt)


# In[50]:


results_dt = pd.DataFrame([results_dt])
results_rf = pd.DataFrame([results_rf])
results_dt['classifier'] = 'Decision Tree'
results_rf['classifier'] = 'Random Forest'


# In[51]:


results_dt


# In[52]:


results_rf


# In[53]:


# Define the classifier
#dt_classifier = DecisionTreeClassifier(random_state=0)
#rf_classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy')

#dts_classifier = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=2)

# Fit the Decision Tree classifier without SMOTE
dt_classifier.fit(X_train, y_train)

# Fit the Random Forest classifier without SMOTE
rf_classifier.fit(X_train, y_train)

# Fit the Decision Tree classifier with SMOTE
dts_classifier.fit(X_train_res, y_train_res)

# Fit the Random Forest classifier with SMOTE
rfs_classifier.fit(X_train_res, y_train_res)

# Define the AUC score within cross-validation for Decision Tree
auc_scores_dt = cross_validate(pipeline_dt, X, y, cv=5, scoring='roc_auc')

# Define the AUC score within cross-validation for Random Forest
auc_scores_rf = cross_validate(pipeline_rf, X, y, cv=5, scoring='roc_auc')

# Average AUC scores
avg_auc_dt = np.mean(auc_scores_dt['test_score'])
avg_auc_rf = np.mean(auc_scores_rf['test_score'])

# Add the cross-validated AUC scores to results DataFrame
results_cv = pd.DataFrame({
    'Model': ['Decision Tree CV with SMOTE', 'Random Forest CV with SMOTE'],
    'CV AUC': [avg_auc_dt, avg_auc_rf]
})

# Plotting ROC curves for all configurations
plt.figure(figsize=(10, 8))

# Plot ROC for Decision Tree without SMOTE
roc_auc_dt = plot_roc_auc(dt_classifier, X_test, y_test, 'Decision Tree')

# Plot ROC for Random Forest without SMOTE
roc_auc_rf = plot_roc_auc(rf_classifier, X_test, y_test, 'Random Forest')

# Plot ROC for Decision Tree with SMOTE
roc_auc_dt_smote = plot_roc_auc(dts_classifier, X_test_res, y_test_res, 'Decision Tree with SMOTE')

# Plot ROC for Random Forest with SMOTE
roc_auc_rf_smote = plot_roc_auc(rfs_classifier, X_test_res, y_test_res, 'Random Forest with SMOTE')

# Add a line for the average cross-validated AUC for Decision Tree with SMOTE
plt.plot(np.nan, np.nan, ' ', label=f'Decision Tree CV with SMOTE (AUC = {avg_auc_dt:.2f})')

# Add a line for the average cross-validated AUC for Random Forest with SMOTE
plt.plot(np.nan, np.nan, ' ', label=f'Random Forest CV with SMOTE (AUC = {avg_auc_rf:.2f})')

# Plot diagonal
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Combine the results DataFrame with the cross-validation results
results_combined = pd.concat([results, results_cv], ignore_index=True, sort=False)
results_combined.fillna('', inplace=True)  # Fill NaN values for neatness

# Display the combined results
print(results_combined)


# ### Checking the number of Dimension

# In[54]:


# Verify the shape of the data
print("X_train shape:", X_train.shape)  # Should be (number_of_samples, 29)
print("X_test shape:", X_test.shape)    # Should be (number_of_samples, 29)

# Correct the input dimension if necessary
# If the number of features is actually 30, change the `input_dim` to 30 as follows:
#classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=30))


# ### ANN: Artificial Neural Network

# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
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

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=29))
classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=30))

# Adding the second hidden layer
classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)  # Set verbose to 0 to avoid excessive output

# Predict on the test data
y_pred_ann = classifier.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5)  # Threshold predictions at 0.5

# Calculate test metrics for ANN
test_accuracy_ann = accuracy_score(y_test, y_pred_ann)
test_precision_ann = precision_score(y_test, y_pred_ann)
test_recall_ann = recall_score(y_test, y_pred_ann)
test_f1_ann = f1_score(y_test, y_pred_ann)

# Compute ROC curve for ANN
fpr_ann, tpr_ann, thresholds_ann = roc_curve(y_test, y_pred_ann)

# Plot ROC curves for all configurations
plt.figure(figsize=(10, 8))

# Plot ROC for Decision Tree without SMOTE
roc_auc_dt = plot_roc_auc(dt_classifier, X_test, y_test, 'Decision Tree')
# Plot ROC for Random Forest without SMOTE
roc_auc_rf = plot_roc_auc(rf_classifier, X_test, y_test, 'Random Forest')
# Plot ROC for Decision Tree with SMOTE
roc_auc_dt_smote = plot_roc_auc(dts_classifier, X_test, y_test, 'Decision Tree with SMOTE')
# Plot ROC for Random Forest with SMOTE
roc_auc_rf_smote = plot_roc_auc(rfs_classifier, X_test, y_test, 'Random Forest with SMOTE')

# Plot ROC for the current model (ANN)
plt.plot(fpr_ann, tpr_ann, label='ANN')

# Add diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()

# Calculate TNR and FNR
tnr = tn / (tn + fp)
fnr = fn / (tp + fn)

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([{
    'Model': 'ANN',
    'Accuracy': test_accuracy_ann,
    'Precision': test_precision_ann,
    'Recall': test_recall_ann,
    'F1 Score': test_f1_ann,
    'TPR': tp / (tp + fn),  # This is the same as recall
    'FPR': fp / (fp + tn),
    'TNR': tnr,
    'FNR': fnr
}])

# Append new row to the DataFrame
results = pd.concat([results, new_row_df], ignore_index=True)

# Display the combined results
print(results)


# # FOR TPU

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 
# # Function to plot ROC AUC for Keras models or any classifier with predict method
# def plot_roc_auc(pred_func, X_test, y_test, model_name):
#     # Calculate the probabilities of each class
#     y_prob = pred_func(X_test)[:, 1] if model_name != 'ANN' else pred_func(X_test).ravel()
# 
#     # Calculate ROC AUC
#     roc_auc = roc_auc_score(y_test, y_prob)
# 
#     # Calculate ROC Curve
#     fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# 
#     # Plot ROC Curve
#     plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
# 
#     return roc_auc
# 
# # Initialize TPU Strategy
# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
#     print('Device:', tpu.master())
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
# except ValueError:  # If not connected to TPU runtime
#     tpu = None
#     strategy = tf.distribute.get_strategy()  # Default strategy that works on CPU and single GPU
# 
# with strategy.scope():
#     # Create and compile your model inside the strategy scope
#     classifier = Sequential()
#     classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=30))
#     classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#     classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# # Assume X_train, y_train, X_test, y_test, dt_classifier, rf_classifier, dts_classifier, and rfs_classifier are available
# 
# # Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
# 
# # Predicting on the test set
# y_pred_ann = classifier.predict(X_test, batch_size=32)
# y_pred_ann = (y_pred_ann > 0.5)
# 
# # Calculate test metrics for ANN
# test_accuracy_ann = accuracy_score(y_test, y_pred_ann)
# test_precision_ann = precision_score(y_test, y_pred_ann)
# test_recall_ann = recall_score(y_test, y_pred_ann)
# test_f1_ann = f1_score(y_test, y_pred_ann)
# 
# # Plot ROC curves for all configurations
# plt.figure(figsize=(10, 8))
# 
# # Assuming that dt_classifier, rf_classifier, dts_classifier, and rfs_classifier are pre-trained classifier instances
# # You need to replace predict_proba with predict for Keras models, as shown below:
# # Note: The dt_classifier, rf_classifier, dts_classifier, and rfs_classifier should provide a predict_proba method
# # if they are non-Keras models. Otherwise, adapt the code accordingly.
# 
# roc_auc_dt = plot_roc_auc(dt_classifier.predict_proba, X_test, y_test, 'Decision Tree')
# roc_auc_rf = plot_roc_auc(rf_classifier.predict_proba, X_test, y_test, 'Random Forest')
# roc_auc_dt_smote = plot_roc_auc(dts_classifier.predict_proba, X_test, y_test, 'Decision Tree with SMOTE')
# roc_auc_rf_smote = plot_roc_auc(rfs_classifier.predict_proba, X_test, y_test, 'Random Forest with SMOTE')
# roc_auc_ann = plot_roc_auc(classifier.predict, X_test, y_test, 'ANN')  # For ANN, use the predict method
# 
# # Add legend and plot formatting
# plt.plot([0, 1], [0, 1], 'k--', label='Random (area = 0.50)')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate (Recall)')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()
# 

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
# 
# # Assuming y_test and y_pred_ann are available and are numpy arrays
# # If y_pred_ann is not a numpy array, convert it using y_pred_ann.numpy() assuming y_pred_ann is a tf.Tensor
# 
# # Calculate the confusion matrix
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
# 
# # Calculate TNR (Specificity) and FNR (Miss rate)
# tnr = tn / (tn + fp)
# fnr = fn / (tp + fn)
# 
# # Assuming test_accuracy_ann, test_precision_ann, test_recall_ann, and test_f1_ann are already calculated using sklearn's functions
# 
# # Create a DataFrame for the new row
# new_row_df = pd.DataFrame([{
#     'Model': 'ANN',
#     'Accuracy': test_accuracy_ann,
#     'Precision': test_precision_ann,
#     'Recall': test_recall_ann,
#     'F1 Score': test_f1_ann,
#     'TPR': tp / (tp + fn),  # This is the same as recall
#     'FPR': fp / (fp + tn),
#     'TNR': tnr,
#     'FNR': fnr
# }])
# 
# # Append new row to the existing DataFrame, ensuring 'results' DataFrame exists
# if 'results' in globals():
#     results = pd.concat([results, new_row_df], ignore_index=True)
# else:
#     results = new_row_df  # If 'results' doesn't exist, initialize it with the new row
# 
# # Display the combined results
# print(results)
# 

# ### Importing The Random Forest with SMOTE Cross Validation Model Into Pickle file

# In[ ]:


### Importing to Pickle file

import pickle

# Assuming rf_classifier is your trained Random Forest model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rfs_classifier, model_file)


# In[ ]:


import pickle
from keras.models import Sequential

# Assuming you have already trained the 'classifier' ANN model
# Save the trained ANN model to a pickle file
with open('ann_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)


# In[ ]:




