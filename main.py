import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.models import Sequential

yf.pdr_override()

moeda_crypto = 'BTC'
moeda_comparativa = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = yf.download('BTC-USD', start, end)

# Data dataframe
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

dias_predicao = 60

x_train, y_train = [], []

for x in range(dias_predicao, len(scaled_data)):
    x_train.append(scaled_data[x-dias_predicao:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Criacao rede neural
print("Shape de x_train:", x_train.shape)
print("Shape de y_train:", y_train.shape)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


#Teste da Model
teste_inicio = dt.datetime(2020,1,1)
teste_fim = dt.datetime.now()

teste_data = yf.download('BTC-USD', start, end)
valor_moeda = teste_data['Close'].values

dataset_total = pd.concat((data['Close'], teste_data['Close']), axis=0)

model_inputs = dataset_total[len(dataset_total) - len(teste_data) - dias_predicao:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_teste = []

for x in range(dias_predicao, len(model_inputs)):
    x_teste.append(model_inputs[x-dias_predicao:x, 0])

x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))

precos_predicao = model.predict(x_teste)
precos_predicao = scaler.inverse_transform(precos_predicao)

plt.plot(valor_moeda, color='black', label='Valores da moeda')
plt.plot(precos_predicao, color='blue', label='Predição de preços')
plt.title(f'Predição do valor da {moeda_crypto}')
plt.xlabel('Time')
plt.ylabel('Preço')
plt.legend(loc='upper left')
plt.show()

