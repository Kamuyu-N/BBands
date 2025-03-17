import pandas as pd
import pandas_ta as ta
from assistance_functions import DeepLearning
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.python.keras.saving.save import load_model

def create_data():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price = pd.read_csv('C:/Users/muyu2/OneDrive/Documents/DeepLearning/aug-sept-2024.csv', sep='\t')
    # price = price[:1000]
    # price = price[-round(len(price)/2):]

    print(len(price))

    price.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tick','volume', 'spread']
    price['datetime'] = price['date'] + " " + price['time']
    price.drop(['date', 'time'], axis=1, inplace=True)

    # Change column arrangement
    price = price[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    price.drop('volume', axis=1, inplace=True)

    # Create log changes for HLC
    log_returns = np.log(price[['high', 'low', 'close', 'open']] / price[['high', 'low', 'close', 'open']].shift(1))
    log_returns.columns = ['log_high', 'log_low', 'log_close', 'log_open']
    log_returns[log_returns.columns] = log_returns[log_returns.columns] * 10000

    # log_returns[log_returns.columns] = scaler.fit_transform(log_returns[log_returns.columns])
    price = pd.concat([price, log_returns], axis=1)

    price['rsi'] = ta.rsi(price.close, 14)
    price['mom'] = ta.mom(price.close, 14)
    price['ema_50'] = ta.ema(price.close, length=50)
    price['ema_200'] = ta.ema(price.close, length=200)
    price['atr'] = ta.atr(price.high, price.low, price.close, length=14)
    bbands = ta.bbands(close=price.close, length=10, mamode='ema')
    price = pd.concat([price, bbands], axis=1)

    price['rsi'] = price['rsi'] / 100
    price['ema_short'] = (price['ema_50'] - price['close']) * 1000
    price['ema_long'] = (price['ema_200'] - price['close']) * 1000

    price.dropna(inplace=True)
    time_step, label = DeepLearning.price_gen_V2(0.0010, 0.0007, price, 8, 20,
                                                 ['rsi', 'log_high', 'log_low', 'log_close', 'ema_long', 'ema_short'])

    df = pd.DataFrame()
    df['timestep'] = time_step
    df['TRD_Outcome'] = label
    print(df)
    print(df['TRD_Outcome'].value_counts())

    # df['TRD_Outcome'] = DeepLearning.label_encode(df.TRD_Outcome)

    np.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/tp_7_sl__6/timesteps.npy", np.array(df.timestep))
    np.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/tp_7_sl__6/label.npy", df.TRD_Outcome)

# create_data()

x1_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/tp_7_sl__6/timesteps.npy', allow_pickle=True)
y_test = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/tp_7_sl__6/label.npy', allow_pickle=True)

tech = np.concatenate(x1_t).flatten()
x1_test= tech.reshape(len(x1_t), 21,6).astype(np.float32)
model = load_model("C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/cluster_bbands.h5")

y_pred = model.predict(x1_test)
Y_classes = np.argmax(y_pred, axis=1)
print(Y_classes)

new_preds = []
buy_thresh, sell_thresh = 0.9 ,0.9
for i in range(0,len(Y_classes)):
        max_index = Y_classes[i]
        if max_index == 1 and y_pred[i][1] >= sell_thresh : # for selling
            new_preds.append(1)
        elif max_index == 0 and y_pred[i][0] >= buy_thresh:
            new_preds.append(0)
        else:
            new_preds.append(2)
print(classification_report(y_test,new_preds))

