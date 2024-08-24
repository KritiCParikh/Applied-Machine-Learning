# -*- coding: utf-8 -*-

!pip install tensorflow
!pip install keras

!pip install protobuf==3.20

!export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import protobuf

import tensorflow as tf

!pip install scipy==1.7

!pip install numpy --upgrade

!pip install dask

"""### Importing Packages"""

import tensorflow

!pip install protoc >= 3.19.0

import pandas as pd
import numpy as np
import xgboost as xgb
import time as t
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

"""### Reading CSVs"""

train = pd.read_csv("train_data.csv")
train.head()



train.shape

labels = pd.read_csv("train_labels.csv")
labels.head()

train['S_2'] = pd.to_datetime(train['S_2'])

df = pd.merge(train, labels, on = 'customer_ID', how = 'left')
df.head()

df.shape

len(df['customer_ID'].unique())

df.loc[df['customer_ID'] == df.iloc[1,0], :].shape

5531451/458913

"""### Randomly Selecting 1 month of data"""

df12 = df.sample(frac=1, random_state = 21)  #shuffling the data
df1 = df12.groupby('customer_ID',as_index=False).first()
print("Min :", df1['S_2'].min(), "Max: ", df1['S_2'].max())
df1.shape

df1.set_index('S_2').groupby(pd.Grouper(freq = 'M')).count()

df1.isnull().sum()

for i in df1.columns:
    if df1[i].dtypes != 'float64':
        print(i, df1[i].dtypes)

df1['D_63'].unique()

df1['D_64'].unique()

"""#### Converting Categorical Variables to dummy variables"""

print("Min: ", df1['S_2'].min(), "Max: ", df1['S_2'].max())

df_new1 = pd.get_dummies(df1, columns = ['D_63', 'D_64'], drop_first = True)

df_new1.shape

df_new1.to_csv('final_df.csv', index = False)

help(pd.read_csv)

df_new = pd.read_csv('final_df.csv', index_col = False)
df_new.shape

df_new.head()

df_new['S_2'] = pd.to_datetime(df_new['S_2'])

df_new[['S_2', 'customer_ID', 'target']].set_index('S_2').groupby(pd.Grouper(freq = 'M')).agg({'customer_ID': 'count', 'target': 'mean'})

for i in df_new.columns:
    print(i)

"""### Split Dataset in Train, Test1 & Test2"""

df_new['S_2'].min()

train_df = df_new.loc[(df_new['S_2'] >= '2017-05-01') & (df_new['S_2'] <= '2018-01-31'), :]
test1_df = df_new.loc[(df_new['S_2'] <= '2017-04-30') , :]
test2_df = df_new.loc[ (df_new['S_2'] >= '2018-02-01'), :]

train_df.shape

test1_df.shape

test2_df.shape

308200 + 61963 + 88750

269243 +30730 +88750

train_df['S_2'].max()

xtrain = train_df.drop(columns = ['target', 'customer_ID', 'S_2'])
ytrain = train_df['target']

xtrain.shape

mm = xgb.XGBClassifier(random_state = 21, scale_pos_weight = 0.5)
mm.fit(xtest1, ytest1)

fi = pd.DataFrame({'columns': xtest1.columns, 'feat_imp': mm.feature_importances_})
fi.loc[fi['feat_imp'] > 0.02,:].sort_values(['feat_imp'], ascending = False)

t1 = t.time()
m1 = xgb.XGBClassifier(random_state = 21)
m1.fit(xtrain, ytrain)
feat_imp = pd.DataFrame({'columns': xtrain.columns, 'feat_imp': m1.feature_importances_})
feat_imp.loc[feat_imp['feat_imp'] > 0.005,:].sort_values(['feat_imp'], ascending = False)
feat_imp.to_csv('feat_imp.csv')
t2 = t.time()
print(t2-t1)

print(m1.learning_rate)

feat_imp = pd.read_csv('feat_imp.csv')
feat_imp.drop(columns = ['Unnamed: 0'], inplace = True)
feat_imp.shape

feat_imp.sort_values(['feat_imp'], ascending = False, inplace = True)
feat_imp.shape

help(plt.xlabel)

plt.figure(figsize=(15,6))
plt.barh(feat_imp.iloc[ :10, 0][::-1], feat_imp.iloc[:10, 1][::-1])
plt.xlabel("Feature Importance: Model 1 Default Params", fontdict = {'family':'sanserif','color':'black','size':15} )

help(xgb.XGBClassifier())

t1 = t.time()
m2 = xgb.XGBClassifier(n_estimators = 300, learning_rate = 0.5,
                       max_depth = 4, subsample = .5, colsample_bytree = 0.5, scale_pos_weight = 5, random_state =21 )
m2.fit(xtrain, ytrain)
feat_imp1 = pd.DataFrame({'columns': xtrain.columns, 'feat_imp': m2.feature_importances_})
feat_imp1.loc[feat_imp1['feat_imp'] > 0.005,:].sort_values(['feat_imp'], ascending = False)
feat_imp1.to_csv('feat_imp1.csv')
t2 = t.time()
print(t2-t1)

feat_imp1.sort_values(['feat_imp'], ascending = False, inplace = True)

feat_imp1.shape

xtrain.shape

sort = m2.feature_importances_.argsort()
sort[:20]

feat_imp1.iloc[ :10, 0]

plt.figure(figsize=(15,6))
plt.barh(feat_imp1.iloc[ :10, 0][::-1], feat_imp1.iloc[:10, 1][::-1])
plt.xlabel("Feature Importance: Model 2 custom Params", fontdict = {'family':'sanserif','color':'black','size':15} )

start = t.time()
df_rnd1 = train.loc[(train['S_2'].dt.year == 2017) & (train['S_2'].dt.month == 7), :]
print(df_rnd1.shape)
end = t.time()
print(end - start)

feat_imp1.loc[feat_imp1['feat_imp'] > 0.005, 'columns'].reset_index()

feat_imp.loc[feat_imp1['feat_imp'] > 0.005, 'columns'].reset_index()

col = list(set(feat_imp1.loc[feat_imp1['feat_imp'] > 0.005, 'columns'].to_list()).union(set(feat_imp.loc[feat_imp['feat_imp'] > 0.005, 'columns'].to_list())))
print(col)

df_new.shape

features = pd.DataFrame(columns = ['Category', '# of Features', '# Selected'])
features['Category'] = ['Delinquency', 'Spend', 'Payment', 'Balance', 'Risk']
features['# of Features'] = 0
features['# Selected'] = 0

for i in df_new.columns:
    print(i)
    if 'P' in i:
        print(i)
        features.loc[features['Category'] == 'Payment', '# of Features'] += 1
    elif 'S' in i:
        features.loc[features['Category'] == 'Spend', '# of Features'] += 1
    elif 'D' in i:
        features.loc[features['Category'] == 'Delinquency', '# of Features'] += 1
    elif 'B' in i:
        features.loc[features['Category'] == 'Balance', '# of Features'] += 1
    else:
        features.loc[features['Category'] == 'Risk', '# of Features'] += 1

for i in col:
    print(i)
    if 'P' in i:
        print(i)
        features.loc[features['Category'] == 'Payment', '# Selected'] += 1
    elif 'S' in i:
        features.loc[features['Category'] == 'Spend', '# Selected'] += 1
    elif 'D' in i:
        features.loc[features['Category'] == 'Delinquency', '# Selected'] += 1
    elif 'B' in i:
        features.loc[features['Category'] == 'Balance', '# Selected'] += 1
    else:
        features.loc[features['Category'] == 'Risk', '# Selected'] += 1

features

features

len(col)

df = df_new.loc[:, col]
df.shape

train_df.shape

xtrain = train_df[col]
xtest1 = test1_df[col]
xtest2 = test2_df[col]

ytrain = train_df['target']
ytest1 = test1_df['target']
ytest2 = test2_df['target']

xtrain.to_csv('xtrain.csv', index = False)
xtest1.to_csv('xtest1.csv', index = False)
xtest2.to_csv('xtest2.csv', index = False)
ytest1.to_csv('ytest1.csv', index = False)
ytest2.to_csv('ytest2.csv', index = False)
ytrain.to_csv('ytrain.csv', index = False)

test2_df.columns

xtrain.shape

ytrain.shape

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

grid_search.sort_values(['AUC Train'], ascending = False)

grid_search.iloc[0, -3]

grid_search.shape

xtrain.shape

xgb_inst = xgb.XGBClassifier(n_estimators = 300, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 1, scale_pos_weight = 1, random_state = 21)
final_xgb = xgb_inst.fit(xtrain, ytrain)
final_xgb.save_model("final_xgb.json")

xtest1 = pd.read_csv('xtest1.csv', index_col = False)
xtest2 = pd.read_csv('xtest2.csv', index_col = False)
ytest1 = pd.read_csv('ytest1.csv', index_col = False)
ytest2 = pd.read_csv('ytest2.csv', index_col = False)
ytrain = pd.read_csv('ytrain.csv', index_col = False)
xtrain = pd.read_csv('xtrain.csv', index_col = False)

final_xgb = xgb.XGBClassifier()
final_xgb.load_model("final_xgb.json")

y_pred = final_xgb.predict_proba(xtrain)[:,1]
ytest1_pred = final_xgb.predict_proba(xtest1)[:,1]
ytest2_pred = final_xgb.predict_proba(xtest2)[:,1]

ytest2_pred

ytrain.

# Rank Ordering
perf_train_data = pd.DataFrame({"Actual": ytrain['target'], "Prediction": final_xgb.predict_proba(xtrain)[:,1]})
quantiles = list(set(perf_train_data.Prediction.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
quantiles.sort()
quantiles.insert(0,0)
quantiles.insert(len(quantiles),1)

quantiles

perf_train_data["Score Bins"] = pd.cut(perf_train_data["Prediction"], quantiles)
stat = perf_train_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat["Bad Rate"] = stat["sum"] / stat["count"]
stat

stat.loc[:, 'Bad Rate'].plot(kind = 'bar', figsize=(15, 8), title = 'Train', fontdict = {'size': 11})

import matplotlib.pyplot as plt

plt.plot(stat["Bad Rate"])

perf_test_data = pd.DataFrame({"Actual": ytest1['target'], "Prediction": final_xgb.predict_proba(xtest1)[:,1]})

perf_test_data["Score Bins"] = pd.cut(perf_test_data["Prediction"], quantiles)
stat1 = perf_test_data.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat1["Bad Rate"] = stat1["sum"] / stat1["count"]
stat1

stat1.loc[:, 'Bad Rate'].plot(kind = 'bar', figsize=(15, 8), title = "Test1")

perf_test_data2 = pd.DataFrame({"Actual": ytest2['target'], "Prediction": final_xgb.predict_proba(xtest2)[:,1]})

perf_test_data2["Score Bins"] = pd.cut(perf_test_data2["Prediction"], quantiles)
stat2 = perf_test_data2.groupby("Score Bins")["Actual"].agg(["sum", "count"])
stat2["Bad Rate"] = stat2["sum"] / stat2["count"]
stat2

stat2.loc[:, 'Bad Rate'].plot(kind = 'bar', figsize=(15, 8), title = "Test2")

ytest2_pred = final_xgb.predict_proba(xtest2)

ytest2_pred[:, 1]

"""### Neural Network

### Normalisation
"""

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

"""### Outlier Treatment"""

xtrain_n_df.describe(percentiles = [.01, .99]).T

xtrain_n_df.quantile(0.99)

for i in xtrain_n_df.columns:
    xtrain_n_df[i] = np.where(xtrain_n_df[i] > xtrain_n_df[i].quantile(0.99), xtrain_n_df[i].quantile(0.99), xtrain_n_df[i] )
    xtrain_n_df[i] = np.where(xtrain_n_df[i] < xtrain_n_df[i].quantile(0.01), xtrain_n_df[i].quantile(0.01), xtrain_n_df[i] )

for i in xtest1_n_df.columns:
    xtest1_n_df[i] = np.where(xtest1_n_df[i] > xtest1_n_df[i].quantile(0.99), xtest1_n_df[i].quantile(0.99), xtest1_n_df[i] )
    xtest1_n_df[i] = np.where(xtest1_n_df[i] < xtest1_n_df[i].quantile(0.01), xtest1_n_df[i].quantile(0.01), xtest1_n_df[i] )

for i in xtest2_n_df.columns:
    xtest2_n_df[i] = np.where(xtest2_n_df[i] > xtest2_n_df[i].quantile(0.99), xtest2_n_df[i].quantile(0.99), xtest2_n_df[i] )
    xtest2_n_df[i] = np.where(xtest2_n_df[i] < xtest2_n_df[i].quantile(0.01), xtest2_n_df[i].quantile(0.01), xtest2_n_df[i] )

xtrain_n_df.describe(percentiles = [.01, .99]).T

"""### Missing Value Imputation"""

xtrain_n_df.fillna(0,inplace=True)
xtest1_n_df.fillna(0,inplace=True)
xtest2_n_df.fillna(0,inplace=True)

def strategy(x, actual, pred, threshold, balance = 'B_2', spend = 'S_3'):
  ydf = pd.DataFrame({'actual': actual, 'pred': pred})
  df = pd.concat([x, ydf], axis = 1)
  total = df.loc[df['pred'] < threshold].shape[0]
  default = df.loc[df['pred'] < threshold, 'actual'].mean()

  df['revenue'] = df.apply(lambda x: x[balance]*0.02 + x[spend]*.001 if x['actual'] != 1 else 0, axis = 1)

  revenue = df.loc[df['pred'] < threshold, 'revenue'].sum()

  return [total, default, revenue]

def ex_sum(xdf, actual, pred, balance = 'B_2', spend = 'S_3'):
    summary = pd.DataFrame(columns = ['Threshold', '#Total', 'Default Rate', 'Revenue'])
    summary['Threshold'] = [(i+1)*0.1 for i in range(10)]
    for i in range(10):
        summary.iloc[i, 1:] = strategy(xdf, actual, pred, summary.iloc[i, 0] , balance = 'B_2', spend = 'S_3')

    return summary

summary = pd.DataFrame(columns = ['Threshold', '#Total', 'Default Rate', 'Revenue'])
for i in range(10):
    threshold =  (i+1)*0.1
    print(threshold)
    summary.iloc[i, 0] = threshold
    summary.iloc[i, 1:] = [2,3,4]

summary.iloc[0, 0] = 3903
summary

summ = ex_sum(xtrain, ytrain['target'], y_pred, balance = 'B_2', spend = 'S_3')
summ

summ1 = ex_sum(xtest1, ytest1['target'], ytest1_pred, balance = 'B_2', spend = 'S_3')
summ1

summ2 = ex_sum(xtest2, ytest2['target'], ytest2_pred, balance = 'B_2', spend = 'S_3')
summ2

x = pd.concat([xtest1, xtrain, xtest2], axis = 0)
y = pd.concat([ytest1, ytrain, ytest2], axis = 0)
yhat = np.concatenate((ytest1_pred , y_pred, ytest2_pred), axis = 0)

overall = ex_sum(x, y['target'], yhat, balance = 'B_2', spend = 'S_3')
overall

help(pd.merge)

summ.merge(summ1, on = 'Threshold', suffixes=('_', ''), how = 'left' ).merge(summ2, on = 'Threshold', how = 'left').merge(overall, on = 'Threshold', how = 'left').to_csv('strategy.csv', index = False)

strategy(xtrain, ytrain['target'], y_pred, 0.55, balance = 'B_2', spend = 'S_3')

def ex1(xdf, actual, pred, c, a, balance = 'B_2', spend = 'S_3'):
    summary = pd.DataFrame(columns = ['#Total', 'Default Rate', 'Revenue'], index = ['Conservative', 'Aggressive'])
    summary.loc['Conservative', :] = strategy(xdf, actual, pred, c, balance = 'B_2', spend = 'S_3')
    summary.loc['Aggressive', :] = strategy(xdf, actual, pred, a, balance = 'B_2', spend = 'S_3')
    return summary

exsum = ex1(xtrain, ytrain['target'], y_pred, 0.5, 0.3, balance = 'B_2', spend = 'S_3')
exsum1 = ex1(xtest1, ytest1['target'], ytest1_pred, 0.5, 0.3, balance = 'B_2', spend = 'S_3')
exsum2 = ex1(xtest2, ytest2['target'], ytest2_pred, 0.5, 0.3, balance = 'B_2', spend = 'S_3')
ex_ov = ex1(x, y['target'], yhat, 0.55, 0.4, balance = 'B_2', spend = 'S_3')

help(df.join)

exsum.join(exsum1, rsuffix='_1').join(exsum2, rsuffix = '_2').join(ex_ov, rsuffix = '_').to_excel('ex_strategy.xlsx')

