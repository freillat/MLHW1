import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('laptops.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
base = ['ram','storage','screen']

print(df.isna().sum())
print(df['ram'].median())

np.random.seed(42)
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
idx = np.arange(n)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train['final_price'].values
y_val = df_val['final_price'].values
y_test = df_test['final_price'].values
del df_train['final_price']
del df_val['final_price']
del df_test['final_price']

def train_linear_regression(X,y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y_train)
    return w_full[0], w_full[1:]

def train_linear_regression_reg(X,y,r=0.01):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y_train)
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def prediction(w0,w,X):
    y_pred = w0 + X.dot(w)
    return y_pred

def prepare_X(df, filler):
    df_num = df[base]
    df_num = df_num.fillna(filler)
    return df_num.values

mean_screen = df_train['screen'].mean()

X_train = prepare_X(df_train, 0)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val, 0)
print(round(rmse(y_val, prediction(w0,w,X_val)),2))

X_train = prepare_X(df_train, mean_screen)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val, mean_screen)
print(round(rmse(y_val, prediction(w0,w,X_val)),2))

X_train = prepare_X(df_train, 0)
X_val = prepare_X(df_val, 0)

r_list = [0, 0.01, 0.1, 1, 5, 10, 100]
for r in r_list:
    w0, w = train_linear_regression_reg(X_train, y_train, r)
    print(r, round(rmse(y_val, prediction(w0, w, X_val)), 2))

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_list = []
for s in seed_list:
    np.random.seed(s)
    idx = np.arange(n)
    np.random.shuffle(idx)
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train = df_train['final_price'].values
    y_val = df_val['final_price'].values
    y_test = df_test['final_price'].values
    del df_train['final_price']
    del df_val['final_price']
    del df_test['final_price']
    X_train = prepare_X(df_train, 0)
    X_val = prepare_X(df_val,0)
    w0, w = train_linear_regression(X_train, y_train)
    rmse_list.append(rmse(y_val, prediction(w0, w, X_val)))
print(round(np.std(rmse_list),3))

np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train['final_price'].values
y_test = df_test['final_price'].values
del df_train['final_price']
del df_test['final_price']
X_train = prepare_X(df_train, 0)
X_test = prepare_X(df_test, 0)
w0, w = train_linear_regression_reg(X_train, y_train, 0.001)
print(rmse(y_test, prediction(w0, w, X_test)))