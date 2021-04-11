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

# + id="AjnNx60adMo5"
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import tensorflow as tf
from pickle import dump
# %matplotlib inline

# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="SS6yHZW6dXAd" outputId="4c13872b-61fd-4783-c822-21b835f1c3f3"
df = pd.read_json('content/btc_ohlc_daily.json')
df.rename_axis('time', axis=0)
target_col = 'close'
target_size = 3

for i, val in enumerate(df.columns):
    if val == target_col:
        target_idx = i

df.head()


# + id="aDyt_7pOM4jy"
# Splits and shuffles the dataset into train, validation and test sets.

def train_val_test_split(df, valid_percent=0.1, test_percent=0.05):
    split_idx_1 = int((1-valid_percent-test_percent)*len(df))
    split_idx_2 = int(split_idx_1 + len(df)*valid_percent)
    return np.split(df, [split_idx_1, split_idx_2]) 

train_df, validation_df, test_df = train_val_test_split(df)

# + colab={"base_uri": "https://localhost:8080/"} id="EKqEsiHOM4jz" outputId="7448fcf6-303c-47f7-b64b-e835bf4a11a5"
print('train:', train_df.shape)
print('validation:', validation_df.shape)
print('test:', test_df.shape)

# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="_RVTUoyLc5tu" outputId="369f159d-6d09-4a93-9695-c2138e1da9da"
# Normalize values between 0-1 for every feature (each column is divided by column max value).

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler.fit(train_df)
scaler.fit(df)

train_scaled = scaler.transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)

train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.columns,index=train_df.index)
validation_scaled_df = pd.DataFrame(validation_scaled, columns=validation_df.columns,index=validation_df.index)
test_scaled_df = pd.DataFrame(test_scaled, columns=test_df.columns,index=test_df.index)

train_scaled_df.head()


# + colab={"base_uri": "https://localhost:8080/", "height": 428} id="mqgciSpHfoy8" outputId="50be6ac9-ffa2-4c90-840a-158816e35f6c"
def line_plot(line1, line2, line3, label1=None, label2=None, label3=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.plot(line3, label=label3, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    
line_plot(train_df[target_col], validation_df[target_col], test_df[target_col], 'training', 'validation', 'test', title='')


# + id="u3EtleUrgD3l"
def extract_window_data(df, window_len=5, output_size=1):
    window_data = []
    for idx in range(len(df) - window_len - output_size + 1):
        tmp = df[idx: (idx + window_len)].copy()
        window_data.append(tmp.values)
    return np.array(window_data)

def extract_target_data(df_target, window_len, output_size):
    target_data = []
    for idx in range(len(df_target) - window_len - output_size + 1):
        tmp = df_target[idx + window_len : idx + window_len + output_size]
        target_data.append(tmp.values)
    return np.array(target_data)

def prepare_data(df, target_col, window_len=10, output_size=1):
    x = extract_window_data(df, window_len, output_size)
    if output_size == 1:
        y = df[target_col][window_len:].values
    else:
        y = extract_target_data(df[target_col], window_len, output_size)
    return x, y

def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape =(input_data.shape[1],input_data.shape[2])))
    #model.add(LSTM(neurons, return_sequences=True, input_shape =(input_data.shape[1],input_data.shape[2])))
    #model.add(Dropout(dropout))
    #model.add(LSTM(neurons))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


# + id="VWxIw0FCgUUa"
np.random.seed(42)
window_len = 10
lstm_neurons = 100
epochs = 200
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

# + id="xfREDGbRV0Yv"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# + id="aI9RtVclghX0"
#X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df_scaled, target_col, window_len=window_len)

X_train, y_train = prepare_data(train_scaled_df, target_col, window_len=window_len, output_size=target_size)
X_val, y_val = prepare_data(validation_scaled_df, target_col, window_len=window_len, output_size=target_size)
X_test, y_test = prepare_data(test_scaled_df, target_col, window_len=window_len, output_size=target_size)
# -

y_train.shape

# + colab={"base_uri": "https://localhost:8080/"} id="t5ghdGoTDiFc" outputId="8fb5f80e-0585-4e6b-e662-f829ebc5d304"
model = build_lstm_model(X_train, output_size=target_size, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[es, mc])

# + colab={"base_uri": "https://localhost:8080/"} id="nINgEWMMM4j3" outputId="b557d9fa-410b-422c-c815-fe2bf21a894e"
# load the best modell
model = load_model('best_model.h5')

# evaluate the model
train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# + colab={"base_uri": "https://localhost:8080/"} id="7BqGSYWIM4j3" outputId="75730a80-0f56-4dbc-e310-81239d29062d"
X_test.shape


# -

def inverse_scaler(scaler, din, target_idx):
    res_array = []
    for i in range(din.shape[1]):
        arr = np.zeros([din.shape[0], scaler.n_features_in_])
        arr[:, target_idx] = din[:,i]
        res = scaler.inverse_transform(arr)
        res_array.append(res[:,target_idx])
    return np.array(res_array)



# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="8yi9lvtOmRRM" outputId="6c508ea8-c3d6-4cfc-e5f7-e167297bdcdc"
preds = model.predict(X_test).squeeze()

res_array = inverse_scaler(scaler, preds, target_idx)

res_array = np.array(res_array)
# -

np.shape(res_array)

i = 0
test_df[target_col][0:len(test_df)]

fff = np.array(res_array)

# + colab={"base_uri": "https://localhost:8080/", "height": 446} id="vMMtJeVAhDRT" outputId="ba0656ef-35fa-43ca-a2cf-9ef0f99aa2ff"
for i in range(target_size):
    predicted_closing_prices = res_array[i,:]
    actual_closing_prices = test_df[target_col][window_len+i:len(test_df)-target_size+1+i]
    predicted_closing_prices = pd.Series(index=actual_closing_prices.index, data=predicted_closing_prices)

    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(actual_closing_prices, label='actual')
    ax.plot(predicted_closing_prices, label='predicted')
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title('Evaluation', fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 533} id="feZTCAj-M4j6" outputId="4c987ae0-0e02-45f0-fed7-827cee42554e"
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

# + colab={"base_uri": "https://localhost:8080/"} id="DKaLn60zM4j6" outputId="4265b041-9c57-47ad-a0c5-b8e27329b85c"
for layer in model.layers:
    print(layer)


# + colab={"base_uri": "https://localhost:8080/"} id="APAKM4k8M4j6" outputId="be711283-320e-4c73-92f4-5d05afaf9341"
for layer in model.layers:
    print('input', layer.input_shape)
    print('output', layer.output_shape)


# + id="wKaH3EgBM4j7"
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))

# + id="kN0UBcnmM4j7"

