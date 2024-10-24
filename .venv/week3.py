import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df_start = pd.read_csv('bank-full.csv', sep=';')
columns_to_keep = ['age','job','marital','education','balance','housing','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
df = df_start[columns_to_keep]
print(df.education.value_counts())

numerical = ['age','balance','day','duration','campaign','pdays','previous']
print(df[numerical].corr())
# `pdays` and `previous`

df['y']=(df['y']=='yes').astype(int)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
df_train=df_train.reset_index().drop(columns='index')
# df_full_train.reset_index()
df_val = df_val.reset_index().drop(columns='index')
# df_test.reset_index()
y_train=df_train.y.values
# y_full_train=df_full_train.y.values
y_val=df_val.y.values
# y_test=df_test.y.values
del df_train['y']
del df_full_train['y']
del df_val['y']
del df_test['y']
assert 'y' not in df_train.columns

for i in ['contact','education','housing','poutcome']:
    print(i,mutual_info_score(df_train[i], y_train))

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

for c_param in [ 0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(solver='liblinear', C=c_param, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    base=(y_pred == y_val).mean()
    print(c_param, base)
    if c_param==1:
        base_rec = base

removals = [ 'age','balance','marital','previous']

for f in removals:
    train_dicts = df_train.drop(columns=f).to_dict(orient='records')
    val_dicts = df_val.drop(columns=f).to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f,base_rec-(y_pred == y_val).mean())
