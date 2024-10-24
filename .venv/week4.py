import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score, roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

df_start = pd.read_csv('bank-full.csv', sep=';')
columns_to_keep = ['age','job','marital','education','balance','housing','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
df = df_start[columns_to_keep]
# print(df.dtypes)

numerical = ['age','balance','day','duration','campaign','pdays','previous']
categorical = ['job','marital','education','housing','contact','month','poutcome']

df['y']=(df['y']=='yes').astype(int)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train=df_train.reset_index(drop=True)
df_full_train=df_full_train.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
y_train=df_train.y.values
y_full_train=df_full_train.y.values
y_val=df_val.y.values
y_test=df_test.y.values
df_train = df_train.drop('y',axis=1)
df_val = df_val.drop('y',axis=1)
df_test = df_test.drop('y',axis=1)

feature_list = ['balance','day','duration','previous']
for feature in feature_list:
    y_pred = df_train[feature].values
    y_pred = y_pred / y_pred.max()
    if roc_auc_score(y_train,y_pred) >=0.5:
        print('%s %.3f' % (feature, roc_auc_score(y_train,y_pred)))
    else:
        print('%s %.3f' % (feature, roc_auc_score(y_train, -y_pred)))
#duration

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:,1]
print('%.3f' % roc_auc_score(y_val,y_pred))

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
for thresh in range(0,101):
    t = thresh/100
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p+r)
    print(t, p, r, f1)
# interects between 0.26 and 0.27
# max f1 at 0.22


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear',C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.y.values
    y_val = df_val.y.values

    dv, model = train(df_train, y_train, C=1)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print(' %.3f +- %.3f' % (np.mean(scores), np.std(scores)))
# std 0.006

for C in [0.000001, 0.001, 1]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.y.values
        y_val = df_val.y.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C,np.mean(scores), np.std(scores)))
# best C = 1