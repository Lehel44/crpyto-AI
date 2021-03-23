#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import urllib.request
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pickle import load


# In[2]:


url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=100'
response = requests.get(url)


# In[3]:


soup = BeautifulSoup(response.text, "html.parser")


# In[4]:


data = json.loads(str(soup))

df = pd.DataFrame(data['Data']['Data'])
df = df.set_index('time')
df.index = pd.to_datetime(df.index, unit='s')
df = df.drop(['conversionType', 'conversionSymbol'], axis=1 )
target_col = 'close'


# In[5]:


model = load_model('best_model.h5')
scaler = load(open('scaler.pkl', 'rb'))


# In[6]:


window_len = model.input_shape[1]


# In[7]:


scaled = scaler.transform(df)
scaled_df = pd.DataFrame(scaled, columns=df.columns,index=df.index)


# In[8]:


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


# In[9]:


x_test, y_test = prepare_data(scaled_df, target_col, window_len)


# In[10]:


preds = model.predict(x_test).squeeze()

preds_arr = np.zeros([preds.shape[0], 6])
preds_arr[:, 5] = preds

preds = scaler.inverse_transform(preds_arr)

preds_df = pd.DataFrame(preds, columns=df.columns)


# In[11]:


predicted_closing_prices = preds_df[target_col].values
actual_closing_prices = df[target_col][window_len:]
predicted_closing_prices = pd.Series(index=actual_closing_prices.index, data=predicted_closing_prices)


# In[12]:


w = scaled_df[len(df)-window_len:len(df)]
wn = np.array(w)
wn = wn[np.newaxis, :, :]

t = model.predict(wn).squeeze()
t_arr = np.zeros(6)
t_arr[5] = t
t_arr = t_arr[np.newaxis, :]
tomorrow = scaler.inverse_transform(t_arr)
tomorrow = tomorrow[0,5]


# In[13]:


save_data = {'time': df.index[window_len:], 'actual': df[target_col][window_len:], 'preds': predicted_closing_prices}
save_df = pd.DataFrame(data=save_data)
save_df = save_df.set_index('time')


# In[14]:


# In[58]:


dt = save_df.index[-1] - save_df.index[-2]
tomorrow_df = pd.DataFrame([[save_df.index[-1] + dt, tomorrow]], columns=['time', 'preds'])
tomorrow_df = tomorrow_df.set_index('time')


# In[59]:


# In[60]:


save_df = save_df.append(tomorrow_df)


# In[61]:


# In[62]:


save_df.to_json(r'results.json')


# In[ ]:




