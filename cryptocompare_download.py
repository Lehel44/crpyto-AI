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
import time
import math

# +
start_time = 1279324800
dt = 24 * 3600 # one day
now = math.floor(time.time())
batch_size = 2000 # max request size

url = 'https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit='

save_path = 'content/btc_ohlc_daily.json'

# +
t = now
urls = []
while t - batch_size * dt > start_time:
    limit = batch_size
    u = url + str(limit) + '&toTs=' + str(t)
    urls.append(u)
    t = t - (batch_size + 1) * dt

u = url + str(math.floor((t-start_time)/dt)) + '&toTs=' + str(t)
urls.append(u)
# -

urls

df_list = []
for i in urls:
    response = requests.get(i)
    soup = BeautifulSoup(response.text, "html.parser")
    data = json.loads(str(soup))
    df_tmp = pd.DataFrame(data['Data']['Data'])
    df_tmp = df_tmp.set_index('time')
    df_tmp.index = pd.to_datetime(df_tmp.index, unit='s')
    df_tmp = df_tmp.drop(['conversionType', 'conversionSymbol'], axis=1 )
    print(df_tmp.head(5))
    print(df_tmp.tail(5))
    df_list.append(df_tmp)

list.reverse(df_list)
df = pd.concat(df_list)

df

df.to_json(save_path)

df2 = pd.read_json(save_path)

df2.rename_axis('time', axis=0)


