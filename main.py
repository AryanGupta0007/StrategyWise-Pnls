import pandas as pd
import numpy as np
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df1 = pd.read_csv('KB188.csv')
df2 = pd.read_csv('KB311.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['tradedPrice'].notna()]

df['filledQuantity'] = np.where(df['transactionType'] == "SELL", df['filledQuantity']*-1, df['filledQuantity'])
df['amount'] = df['tradedPrice'] * df['filledQuantity']
client_1 = df.loc[df['LoginID'] == 'KB188']
client_2 = df.loc[df['LoginID'] == 'KB311']
client_1_PnL = client_1['amount'].sum()
client_2_PnL = client_2['amount'].sum()
df['orderTimestamp'] = pd.to_datetime(df['orderTimestamp'])
df = df.drop('date', axis=1)
# print(df)
df['date'] = df['orderTimestamp'].dt.date
# print(df)
orderTags = df['orderTag'].unique()
orderTags = list(orderTags)
clients = df['LoginID'].unique()
