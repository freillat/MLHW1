import pandas as pd
import numpy as np

df = pd.read_csv('laptops.csv')

print(df.info())

print(len(df['Brand'].unique()))

print(df.isna().any())

print(df[df['Brand']=='Dell']['Final Price'].max())

print(df['Screen'].median())
screen = df['Screen'].mode()
df_screen = df['Screen'].fillna(screen)
print(df_screen.median())

X = df[df['Brand']=='Innjoo'][['RAM', 'Storage', 'Screen']].to_numpy()
XTX = np.matmul(X.T, X)
invXTX = np.linalg.inv(XTX)
y = [1100, 1300, 800, 900, 1000, 1100]
w = np.matmul(np.matmul(invXTX , X.T), y)
print(w.sum())
