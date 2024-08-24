# -*- coding: utf-8 -*-

### Installing packages using pip
"""

!pip install tensorflow
!pip install keras

|!pip install scipy==1.7

!pip install dask

"""### Importing Packages"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import time as t
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

# fine tuning with Grid Search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)

"""### Reading CSVs

Used data from "American Express - Default Prediction" Kaggle Competition.

The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.
"""

train = pd.read_csv("train_data.csv") # X variables
train.head()

train.shape

labels = pd.read_csv("train_labels.csv") # Y variables
labels.head()

# Convert Date column to datetime() Data Type
train['S_2'] = pd.to_datetime(train['S_2'])

# Merge the Features and Labels (Target values)
df = pd.merge(train, labels, on = 'customer_ID', how = 'left')
df.head()

df.shape #check the shape of the final dataframe

len(df['customer_ID'].unique())

"""TOTAL No of CUSTOMERS - 458913"""

df.loc[df['customer_ID'] == df.iloc[1,0], :].shape

5531451/458913

"""### Randomly Selecting 1 month of data for each customer



"""

df12 = df.sample(frac=1, random_state = 21)  #shuffling the data

#Grouping by the customer ID and selecting the first row
df1 = df12.groupby('customer_ID',as_index=False).first()

#checking the Minimum and Maximum Date for the data
print("Min :", df1['S_2'].min(), "Max: ", df1['S_2'].max())
df1.shape

# Grouping by the Month and counting the rows in each of them
df1.set_index('S_2').groupby(pd.Grouper(freq = 'M')).count()

# Checking Null Values in each columns
# Target column shouldn't have any null value
df1.isnull().sum()

"""Checking for Categorical Variables"""

# Run the loop through each column and check it's datatype
# If Data Type not equal to Float64, then print

for i in df1.columns:
    if df1[i].dtypes != 'float64':
        print(i, df1[i].dtypes)

# Check unique values in D_63 column
df1['D_63'].unique()

# Check unique values in D_64 column
df1['D_64'].unique()

"""#### Converting Categorical Variables to dummy variables"""

print("Min: ", df1['S_2'].min(), "Max: ", df1['S_2'].max())

# fn converts categorical values to dummy values and drops the og columns
df_new1 = pd.get_dummies(df1, columns = ['D_63', 'D_64'], drop_first = True)

df_new1.shape

# Save the final dataframe in csv
df_new1.to_csv('final_df.csv', index = False)

df_new = pd.read_csv('final_df.csv', index_col = False)
df_new.shape

df1.set_index('S_2').groupby(pd.Grouper(freq = 'M')).count()

for i in df_new.columns:
    print(i)

"""### Split Dataset in Train, Test1 & Test2"""

train_df = df_new.loc[(df_new['S_2'] >= '2017-05') & (df_new['S_2'] <= '2018-01'), :]
test1_df = df_new.loc[(df_new['S_2'] <= '2017-04') , :]
test2_df = df_new.loc[ (df_new['S_2'] >= '2018-02'), :]

train_df.shape

xtrain = train_df.drop(columns = ['target', 'customer_ID', 'S_2'])
ytrain = train_df['target']

xtrain.shape

"""### Feature Selection by extracting feature importance from 2 models"""

### MODEL 1
t1 = t.time()
m1 = xgb.XGBClassifier(random_state = 21)
m1.fit(xtrain, ytrain)
feat_imp = pd.DataFrame({'columns': xtrain.columns, 'feat_imp': m1.feature_importances_}) # creating table to rank feature importances
feat_imp.loc[feat_imp['feat_imp'] > 0.005,:].sort_values(['feat_imp'], ascending = False)
feat_imp.to_csv('feat_imp.csv')
t2 = t.time()
print(t2-t1)

# MODEL 2 - with some parameters as defined by professor
t1 = t.time() #max_dept = max number of nodes, subsample = % of rows considered from the total data in the main tree at once, colsample_bytree = % of columns considered from the total data in the main tree at once
m2 = xgb.XGBClassifier(n_estimators = 300, learning_rate = 0.5,
                       max_depth = 4, subsample = .5, colsample_bytree = 0.5, scale_pos_weight = 5, random_state =21 )
m2.fit(xtrain, ytrain)
feat_imp1 = pd.DataFrame({'columns': xtrain.columns, 'feat_imp': m2.feature_importances_})
feat_imp1.loc[feat_imp1['feat_imp'] > 0.005,:].sort_values(['feat_imp'], ascending = False)
feat_imp1.to_csv('feat_imp1.csv')
t2 = t.time()
print(t2-t1)

#the total time taken to run the code
start = t.time()
df_rnd1 = train.loc[(train['S_2'].dt.year == 2017) & (train['S_2'].dt.month == 7), :]
print(df_rnd1.shape)
end = t.time()
print(end - start)

#segregated the features based on their feature importance > 0.005 for model imp1
feat_imp1.loc[feat_imp1['feat_imp'] > 0.005, 'columns'].reset_index()

#segregated the features based on their feature importance > 0.005 for model imp1
feat_imp.loc[feat_imp1['feat_imp'] > 0.005, 'columns'].reset_index()

"""### Final Sets of features to be used for modelling"""

#union the above two chunk's output based on the filter by ignoring values that occur twice. Eg : A B C A D E - here we consider 5 values
col = list(set(feat_imp1.loc[feat_imp1['feat_imp'] > 0.005, 'columns'].to_list()).union(set(feat_imp.loc[feat_imp['feat_imp'] > 0.005, 'columns'].to_list())))
print(col)

len(col)

df = df_new.loc[:, col]
df.shape

train_df.shape

#Refer section "Split Dataset in Train, Test1 & Test2".
#train_df contains 197 columns within the range of dates assigned to it. Now, after we assign the [col] to it, we are only choosing the columns that are in the col variable i.e. there are 44
xtrain = train_df[col]
xtest1 = test1_df[col]
xtest2 = test2_df[col]

#refer the same train_df variable output
ytrain = train_df['target']
ytest1 = test1_df['target']
ytest2 = test2_df['target']

#saving it to csv
xtrain.to_csv('xtrain.csv', index = False)
xtest1.to_csv('xtest1.csv', index = False)
xtest2.to_csv('xtest2.csv', index = False)
ytest1.to_csv('ytest1.csv', index = False)
ytest2.to_csv('ytest2.csv', index = False)
ytrain.to_csv('ytrain.csv', index = False)

test2_df.columns

xtrain.shape

ytrain.shape

"""## XGBoost

### Grid Search for XGBoost model

Combinations:
1. Number of Trees: [50, 100, 300]
2. Learning Rates = [0.01, .1]
3. % of obs used in each tree = [.5, .8]
4. % of features used in each tree = [.5, 1]
5. Weight of default obs = [1, 5, 10]
"""

t1 = t.time()
grid_search = pd.DataFrame(columns = ['No. of Trees', 'LR', 'Subsample', '%features', 'Default Weight', 'AUC Train', 'AUC Test1', 'AUC Test2'])
num_trees = [50, 100, 300]
lr = [0.01, .1]
subsample = [.5, .8]
feat = [.5, 1]
def_w = [1, 5, 10]

row = 0
for i in num_trees:
    for j in lr:
        for k in subsample:
            for f in feat:
                for d in def_w:
                    xgb_inst = xgb.XGBClassifier(n_estimators = i, learning_rate = j, subsample = k, colsample_bytree = f, scale_pos_weight = d, random_state = 21)
                    model = xgb_inst.fit(xtrain, ytrain)
                    print(row)
                    grid_search.loc[row, 'No. of Trees'] = i
                    grid_search.loc[row, 'LR'] = j
                    grid_search.loc[row, 'Subsample'] = k
                    grid_search.loc[row, '%features'] = f
                    grid_search.loc[row, 'Default Weight'] = d
                    grid_search.loc[row,"AUC Train"] = roc_auc_score(ytrain, model.predict_proba(xtrain)[:,1])
                    grid_search.loc[row,"AUC Test1"] = roc_auc_score(ytest1, model.predict_proba(xtest1)[:,1])
                    grid_search.loc[row,"AUC Test2"] = roc_auc_score(ytest2, model.predict_proba(xtest2)[:,1])
                    row += 1

t2 = t.time()
print(t2-t1)
grid_search

grid_search.to_csv("grid_search.csv", index = False)

grid_search.loc[0, '%features']

grid_search.iloc[0, -3]

grid_search.shape

"""### Final XGB MODEL"""

xgb_inst = xgb.XGBClassifier(n_estimators = 300, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 1, scale_pos_weight = 1, random_state = 21)
final_xgb = xgb_inst.fit(xtrain, ytrain)
final_xgb.save_model("final_xgb.json")

roc_auc_score(ytrain, final_xgb.predict_proba(xtrain)[:,1])

"""## Neural Network"""

xtest1 = pd.read_csv('xtest1.csv', index_col = False)
xtest2 = pd.read_csv('xtest2.csv', index_col = False)
ytest1 = pd.read_csv('ytest1.csv', index_col = False)
ytest2 = pd.read_csv('ytest2.csv', index_col = False)
ytrain = pd.read_csv('ytrain.csv', index_col = False)
xtrain = pd.read_csv('xtrain.csv', index_col = False)

print(xtest1.shape, xtest2.shape)

"""### Outlier Treatment"""

xtrain.describe(percentiles = [.01, .99]).T

xtrain.quantile(0.99)

for i in xtrain.columns:
    xtrain[i] = np.where(xtrain[i] > xtrain[i].quantile(0.99), xtrain[i].quantile(0.99), xtrain[i] )
    xtrain[i] = np.where(xtrain[i] < xtrain[i].quantile(0.01), xtrain[i].quantile(0.01), xtrain[i] )

for i in xtest1.columns:
    xtest1[i] = np.where(xtest1[i] > xtest1[i].quantile(0.99), xtest1[i].quantile(0.99), xtest1[i] )
    xtest1[i] = np.where(xtest1[i] < xtest1[i].quantile(0.01), xtest1[i].quantile(0.01), xtest1[i] )

for i in xtest2.columns:
    xtest2[i] = np.where(xtest2[i] > xtest2[i].quantile(0.99), xtest2[i].quantile(0.99), xtest2[i] )
    xtest2[i] = np.where(xtest2[i] < xtest2[i].quantile(0.01), xtest2[i].quantile(0.01), xtest2[i] )

xtrain.describe(percentiles = [.01, .99]).T

"""### Normalisation"""

from sklearn.preprocessing import StandardScaler, RobustScaler
sc = StandardScaler()
sc.fit(xtrain)

xtrain_n = sc.transform(xtrain)
xtest1_n  = sc.transform(xtest1)
xtest2_n  = sc.transform(xtest2)

# convert to Pandas DF
xtrain_n_df = pd.DataFrame(xtrain_n, columns=xtrain.columns)
xtest1_n_df = pd.DataFrame(xtest1_n, columns=xtest1.columns)
xtest2_n_df = pd.DataFrame(xtest2_n, columns=xtest2.columns)

"""### Missing Value Imputation"""

xtrain_n_df.fillna(0,inplace=True)
xtest1_n_df.fillna(0,inplace=True)
xtest2_n_df.fillna(0,inplace=True)

xtest2_n_df.head()

def build_classifier(activation = 'relu', dropout_rate = .5, neurons = 4):
    # first step: create a Sequential object, as a sequence of layers. B/C NN is a sequence of layers.
    classifier = Sequential()
    # add the first hidden layer
    classifier.add(Dense(units=neurons,kernel_initializer='glorot_uniform',
                    activation = activation))
    classifier.add(Dropout(dropout_rate))
    # add the second hidden layer
    classifier.add(Dense(units=neurons,kernel_initializer='glorot_uniform',
                    activation = activation))
    # add the output layer
    classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',
                    activation = 'sigmoid'))
    # compiling the NN
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size = 100)

parameters = dict(activation = ['relu', 'tanh'], batch_size = [100, 10000], )
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'roc_auc', cv=10, return_train_score=True)
grid_search = grid_search.fit(xtrain_n_df, ytrain)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

grid_search.

best_parameters

train_pred = grid_search.predict_proba(xtrain_n_df)

ytrain.sum()

train_pred

roc_auc_score(ytrain, train_pred[:, 1])

test1_pred = grid_search.predict_proba(xtest1_n_df)[:,1]
test2_pred = grid_search.predict_proba(xtest2_n_df)[:,1]

print(roc_auc_score(ytest1, test1_pred))
print(roc_auc_score(ytest2, test2_pred))

best_accuracy

grid_search.cv_results_

# print the AUC scores for all hyperparameters
results = grid_search.cv_results_
for i in range(len(results['params'])):
    print(f"Hyperparameters: {results['params'][i]}")
    #print(f"Train AUC score: {results['mean_train_score'][i]:.3f}")
    print(f"Test AUC score: {results['mean_test_score'][i]:.3f}")
    print()

"""Grid Search sucks!

For 2 hidden Layers
"""

y_pred = classifier.predict(xtest2_n_df)
y_pred

roc_auc_score(ytrain, y_pred)

ytest2['target'].shape

y_pred[:, 0].shape



# first step: create a Sequential object, as a sequence of layers. B/C NN is a sequence of layers.
t1 = t.time()
grid_search_nn1 = pd.DataFrame(columns = ['hd', 'nodes', 'activation', 'dropout', 'batch_size', 'auc_train', 'auc_test1', 'auc_test2'])


neurons = [4, 6]
activations = ['relu', 'tanh']
dropout = [.5, 0]
batch_sizes = [100, 10000]

row = 0
for i in neurons:
  for a in activations:
    for d in dropout:
      for s in batch_sizes:

        grid_search_nn1.loc[row, 'nodes'] = i
        grid_search_nn1.loc[row, 'activation'] = a
        grid_search_nn1.loc[row, 'dropout'] = d
        grid_search_nn1.loc[row, 'batch_size'] = s

        classifier = Sequential()

        # add the first hidden layer
        classifier.add(Dense(units=i,kernel_initializer='glorot_uniform',
                            activation = a))

        classifier.add(Dropout(d))

        # add the second hidden layer
        classifier.add(Dense(units=i,kernel_initializer='glorot_uniform',
                        activation = a))

        classifier.add(Dropout(d))
        # add the output layer
        classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',
                            activation = 'sigmoid'))

        # add additional parameters
        classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', 'FalseNegatives'])

        # train the model
        classifier.fit(xtrain_n_df,ytrain,batch_size=s,epochs=20)

        grid_search_nn1.loc[row, 'auc_train'] = roc_auc_score(ytrain, classifier.predict(xtrain_n_df))
        grid_search_nn1.loc[row, 'auc_test1'] = roc_auc_score(ytest1, classifier.predict(xtest1_n_df))
        grid_search_nn1.loc[row, 'auc_test2'] = roc_auc_score(ytest2, classifier.predict(xtest2_n_df))
        print("Best Parameter Iteration",  row, "AUC Train", roc_auc_score(ytrain, classifier.predict(xtrain_n_df)))

        row += 1

grid_search_nn1['hd'] = 2
t2 = t.time()
print(t2 - t1)

grid_search_nn1['hd'] = 2

grid_search_nn1.to_csv('grid_1.csv', index = False)

grid_search_nn1 = pd.read_csv('grid_1.csv', index_col=False)
grid_search_nn1

# first step: create a Sequential object, as a sequence of layers. B/C NN is a sequence of layers.
t1 = t.time()
grid_search_nn2 = pd.DataFrame(columns = ['hd', 'nodes', 'activation', 'dropout', 'batch_size', 'auc_train', 'auc_test1', 'auc_test2'])


neurons = [4, 6]
activations = ['relu', 'tanh']
dropout = [.5, 0]
batch_sizes = [100, 10000]

row = 0
for i in neurons:
  for a in activations:
    for d in dropout:
      for s in batch_sizes:

        grid_search_nn2.loc[row, 'nodes'] = i
        grid_search_nn2.loc[row, 'activation'] = a
        grid_search_nn2.loc[row, 'dropout'] = d
        grid_search_nn2.loc[row, 'batch_size'] = s

        classifier = Sequential()

        # add the first hidden layer
        classifier.add(Dense(units=i,kernel_initializer='glorot_uniform',
                            activation = a))
        classifier.add(Dropout(d))

        # add the second hidden layer
        classifier.add(Dense(units=i,kernel_initializer='glorot_uniform',
                        activation = a))
        classifier.add(Dropout(d))

        # add the third hidden layer
        classifier.add(Dense(units=i,kernel_initializer='glorot_uniform',
                        activation = a))
        classifier.add(Dropout(d))

        # add the fourth hidden layer
        classifier.add(Dense(units=i,kernel_initializer='glorot_uniform',
                        activation = a))
        classifier.add(Dropout(d))

        # add the output layer
        classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',
                            activation = 'sigmoid'))

        # add additional parameters
        classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', 'FalseNegatives'])

        # train the model
        classifier.fit(xtrain_n_df,ytrain,batch_size=s,epochs=20)

        grid_search_nn2.loc[row, 'auc_train'] = roc_auc_score(ytrain, classifier.predict(xtrain_n_df))
        grid_search_nn2.loc[row, 'auc_test1'] = roc_auc_score(ytest1, classifier.predict(xtest1_n_df))
        grid_search_nn2.loc[row, 'auc_test2'] = roc_auc_score(ytest2, classifier.predict(xtest2_n_df))
        print("Best Parameter Iteration",  row, "AUC Train", roc_auc_score(ytrain, classifier.predict(xtrain_n_df)))

        row += 1

grid_search_nn2['hd'] = 4
t2 = t.time()
print(t2 - t1)

grid_search_nn2

grid_search_nn2.to_csv('grid_2.csv', index = False)

grid_nn = pd.concat([grid_search_nn1, grid_search_nn2], axis=0)
grid_nn.sort_values(['auc_train'], ascending=False)

grid_nn.to_csv('grid_nn.csv', index = False)

xtrain_n_df.shape

"""### Final Neural Network"""

# first step: create a Sequential object, as a sequence of layers. B/C NN is a sequence of layers.
classifier = Sequential()

# add the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',
                    activation = 'relu'))

# add the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='glorot_uniform',
                activation = 'relu'))

# add the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',
                    activation = 'sigmoid'))

# add additional parameters
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', 'FalseNegatives'])

# train the model
classifier.fit(xtrain_n_df,ytrain,batch_size=100,epochs=20)

filename = 'final_nn.sav'
pickle.dump(classifier, open(filename, 'wb'))

roc_auc_score(ytrain, y_pred)

filename = 'final_nn.sav'
classifier = pickle.load(open(filename, 'rb'))

y_pred = classifier.predict(xtrain_n_df)
y_pred



ytest1_pred = classifier.predict(xtest1_n_df)
ytest1_pred

ytest2_pred = classifier.predict(xtest2_n_df)
ytest2_pred

"""## Rank Ordering"""

# Rank Ordering
perf_train_data = pd.DataFrame({"Actual": ytrain['target'], "Prediction":y_pred[:, 0] })
quantiles = list(set(perf_train_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)
quantiles

perf_train_data["Score Bins"] = pd.cut(perf_train_data["Prediction"], quantiles)
stat = perf_train_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat["Bad Rate"] = stat["sum"] / stat["count"]
stat

stat.loc[:, 'Bad Rate'].plot(kind = 'bar', figsize=(15, 8), title = 'Train')

perf_test_data = pd.DataFrame({"Actual": ytest1['target'], "Prediction": ytest1_pred[:,0]})

perf_test_data["Score Bins"] = pd.cut(perf_test_data["Prediction"], quantiles)
stat1 = perf_test_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat1["Bad Rate"] = stat1["sum"] / stat1["count"]
stat1.loc[:, 'Bad Rate'].plot(kind = 'bar', figsize=(15, 8), title = 'Test1')

perf_test_data1 = pd.DataFrame({"Actual": ytest2['target'], "Prediction": ytest2_pred[:,0]})

perf_test_data1["Score Bins"] = pd.cut(perf_test_data1["Prediction"], quantiles)
stat2 = perf_test_data1.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat2["Bad Rate"] = stat2["sum"] / stat2["count"]
stat2.loc[:, 'Bad Rate'].plot(kind = 'bar', figsize=(15, 8), title = 'Test2')

xtest2 = pd.read_csv('xtest2.csv')
ytest2 = pd.read_csv('ytest2.csv')

for i in xtest2.columns:
  if ('B_' in i ) or ('S_' in i ):
    print(i)

"""## Strategy for calculating - Default Rate and Revenue"""

def strategy(x, actual, pred, threshold, balance = 'B_2', spend = 'S_3'):
  ydf = pd.DataFrame({'actual': actual, 'pred': pred})
  df = pd.concat([x, ydf], axis = 1)
  total = df.loc[df['pred'] < threshold].shape[0]
  default = df.loc[df['pred'] < threshold, 'actual'].mean()

  df['revenue'] = df.apply(lambda x: x[balance]*0.02 + x[spend]*.001 if x['actual'] != 1 else 0, axis = 1)

  revenue = df.loc[df['pred'] < threshold, 'revenue'].sum()

  return [total, default, revenue]

strategy(xtrain, ytrain['target'], y_pred[:,0], 0.55, balance = 'B_2', spend = 'S_3')

strategy(xtrain, ytrain['target'], y_pred[:,0], 0.35, balance = 'B_2', spend = 'S_3')

strategy(xtest1, ytest1['target'], ytest1_pred[:,0], 0.55, balance = 'B_2', spend = 'S_3')

threshold = np.linspace(0, 1, 100)

threshold[1]

threshold[1]

r = np.array([])
r = np.append([r],[2])
np.append([r],[2])

#

import matplotlib.pyplot as plt

threshold = np.linspace(0, 1, 100)
default = np.array([])
revenue = np.array([])

for i in range(100) :
    lis = strategy(xtrain, ytrain['target'], y_pred[:,0], threshold[i], balance = 'B_2', spend = 'S_3')
    default = np.append(default, lis[0])
    revenue = np.append(revenue, lis[1])

fig, ax = plt.subplots()

ax.plot(threshold, default, label = 'default')

ax.plot(threshold, revenue, label = 'revenue')

# Add a legend
ax.legend()

# Show the plot
plt.show()

plt.plot(threshold, default, label = 'default')
plt.show()

plt.plot(threshold, revenue, label = 'default')
plt.show()

"""## Executive Summary"""

# Conservative Strategy - Threshold = 0.55
# Aggressive Strategy - Threshold = 0.4



def ex_sum(xdf, actual, pred, c, a, balance = 'B_2', spend = 'S_3'):
    summary = pd.DataFrame(columns = ['#Total', 'Default Rate', 'Revenue'], index = ['Conservative', 'Aggressive'])
    summary.loc['Conservative', :] = strategy(xdf, actual, pred, c, balance = 'B_2', spend = 'S_3')
    summary.loc['Aggressive', :] = strategy(xdf, actual, pred, a, balance = 'B_2', spend = 'S_3')
    return summary

ex_sum(xtrain, ytrain['target'], y_pred[:,0], 0.5, 0.4, balance = 'B_2', spend = 'S_3')

ex_sum(xtest1, ytest1['target'], ytest1_pred[:,0], 0.5, 0.4, balance = 'B_2', spend = 'S_3')

ex_sum(xtest2, ytest2['target'], ytest2_pred[:,0], 0.5, 0.4, balance = 'B_2', spend = 'S_3')

print(xtest2.shape)
print(xtest1.shape)

print(ytest1_pred )

ytrain.shape

yhat = np.concatenate((ytest1_pred , ytrain), axis = 0)
yhat.shape

x = pd.concat([xtest1, xtrain, xtest2], axis = 0)
y = pd.concat([ytest1, ytrain, ytest2], axis = 0)
yhat = np.concatenate((ytest1_pred , y_pred, ytest2_pred), axis = 0)


ex_sum(x, y['target'], yhat[:,0], 0.5, 0.4, balance = 'B_2', spend = 'S_3')

roc_auc_score(ytest1, ytest1_pred[:,0])

ytest1.mean()

ytest2.mean()

ytrain.mean()

while 2 != 0:
    i +=1

i

