# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: crypto-v3.8
#     language: python
#     name: crypto-v3.8
# ---

import requests
import urllib.request
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pickle import load
import matplotlib.pyplot as plt
# %matplotlib inline

url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=100'
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

# +
data = json.loads(str(soup))

df = pd.DataFrame(data['Data']['Data'])
df = df.set_index('time')
df.index = pd.to_datetime(df.index, unit='s')
df = df.drop(['conversionType', 'conversionSymbol'], axis=1 )
target_col = 'close'

df.head()
# -

model = load_model('best_model.h5')
scaler = load(open('scaler.pkl', 'rb'))

window_len = model.input_shape[1]

scaled = scaler.transform(df)
scaled_df = pd.DataFrame(scaled, columns=df.columns,index=df.index)


# +
def extract_window_data(df, window_len=5):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        window_data.append(tmp.values)
    
    return np.array(window_data)

def prepare_data(df, target_col, window_len=10):

    x = extract_window_data(df, window_len)
    y = df[target_col][window_len:].values
    
    return x, y



# -

x_test, y_test = prepare_data(scaled_df, target_col, window_len)

# +
preds = model.predict(x_test).squeeze()

preds_arr = np.zeros([preds.shape[0], 6])
preds_arr[:, 5] = preds

preds = scaler.inverse_transform(preds_arr)

preds_df = pd.DataFrame(preds, columns=df.columns)

preds_df.head()

# +
predicted_closing_prices = preds_df[target_col].values
actual_closing_prices = df[target_col][window_len:]
predicted_closing_prices = pd.Series(index=actual_closing_prices.index, data=predicted_closing_prices)

fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(actual_closing_prices, label='actual')
ax.plot(predicted_closing_prices, label='predicted')
ax.set_ylabel('price [USD]', fontsize=14)
ax.set_title('Evaluation', fontsize=16)
ax.legend(loc='best', fontsize=16)
plt.show()

# +
w = scaled_df[len(df)-window_len:len(df)]
wn = np.array(w)
wn = wn[np.newaxis, :, :]

t = model.predict(wn).squeeze()
t_arr = np.zeros(6)
t_arr[5] = t
t_arr = t_arr[np.newaxis, :]
tomorrow = scaler.inverse_transform(t_arr)
tomorrow = tomorrow[0,5]
# -

save_data = {'time': df.index[window_len:], 'actual': df[target_col][window_len:], 'preds': predicted_closing_prices}
save_df = pd.DataFrame(data=save_data)
save_df = save_df.set_index('time')

save_df

dt = save_df.index[-1] - save_df.index[-2]
tomorrow_df = pd.DataFrame([[save_df.index[-1] + dt, tomorrow]], columns=['time', 'preds'])
tomorrow_df = tomorrow_df.set_index('time')

tomorrow_df

save_df = save_df.append(tomorrow_df)

save_df

save_df.to_json(r'results.json')


