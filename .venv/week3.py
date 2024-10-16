import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# df.columns = df.columns.str.lower().str.replace(' ','_')
# categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
# for c in categorical_columns:
#     df[c] = df[c].str.str.lower().str.replace(' ','_')

df_start = pd.read_csv('bank-full.csv', sep=';')
columns_to_keep = ['age','job','marital','education','balance','housing','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
# columns_to_keep = ['job','marital','education','balance','housing','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
# columns_to_keep = ['age','job','marital','education','housing','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
# columns_to_keep = ['age','job','education','balance','housing','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
# columns_to_keep = ['age','job','marital','education','balance','housing','contact','day','month','duration','campaign','pdays','poutcome','y']
# columns_to_keep = ['age','balance','marital','previous','y']
# 'age',
df = df_start[columns_to_keep]
print(df.education.value_counts())
print(df.dtypes)

numerical = ['age','balance','day','duration','campaign','pdays','previous']
print(df[numerical].corr())

# `pdays` and `previous`

df['y']=(df['y']=='yes').astype(int)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
df_train.reset_index()
df_full_train.reset_index()
df_val.reset_index()
df_test.reset_index()
y_train=df_train.y.values
y_full_train=df_full_train.y.values
y_val=df_val.y.values
y_test=df_test.y.values
del df_train['y']
del df_full_train['y']
del df_val['y']
del df_test['y']

for i in ['contact','education','housing','poutcome']:
    print(i,mutual_info_score(df_train[i], y_train))

train_dicts = df_train.to_dict(orient='records')
# full_train_dicts = df_full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
# X_full_train = dv.fit_transform(full_train_dicts)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)
# test_dicts = df_test.to_dict(orient='records')
# X_test = dv.transform(test_dicts)

model = LogisticRegression(solver='liblinear', C=10.0, max_iter=1000, random_state=42)
# model.fit(X_full_train, y_full_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print((y_pred == y_val).mean().round(3))