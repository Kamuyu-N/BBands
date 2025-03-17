import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from assistance_functions import DeepLearning
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle

# Set a maximum memory limit
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=2046)])

def create_data():
    scaler = MinMaxScaler(feature_range=(-1,1))
    price = pd.read_csv('C:/Users/muyu2/OneDrive/Documents/DeepLearning/eurusd-15m.csv', sep=';')
    # price = price[-round(len(price)/2):]
    print(len(price))

    price.columns = ['date', 'time', 'open', 'high','low','close', 'volume']
    price['datetime'] = price['date'] + " "+ price['time']
    price.drop(['date', 'time'], axis=1, inplace=True)

    #Change column arrangement
    price = price[['datetime', 'open','high','low','close', 'volume']]
    price.drop('volume', axis= 1, inplace=True)

    #Create log changes for HLC
    log_returns = np.log(price[['high','low','close','open']] / price[['high','low','close','open']].shift(1))
    log_returns.columns = ['log_high', 'log_low', 'log_close','log_open']

    log_returns[log_returns.columns] = log_returns[log_returns.columns] *1000
    price = pd.concat([price,log_returns], axis=1)

    price['rsi'] = ta.rsi(price.close, 14)
    price['ema_50'] = ta.ema(price.close, length= 50)
    price['ema_200'] = ta.ema(price.close, length = 200)
    bbands = ta.bbands(close=price.close, length=20, mamode='ema') #Creation of the Indicator
    price = pd.concat([price,bbands], axis=1)

    price['rsi'] = price['rsi']/100
    price['ema_short']  = (price['ema_50'] - price['close']) * 1000
    price['ema_long'] = (price['ema_200'] - price['close']) * 1000
    price.dropna(inplace= True)

    if input("bbands or norm...\n") == 'bbands':
        label, time_step  = DeepLearning.price_gen_V3(price, forward_length=20, timestep_length=40,indicator_columns=['rsi','log_high','log_low','log_close', 'ema_long','ema_short'], repeat=False)

    else:
        time_step, label = DeepLearning.price_gen_V2(0.0010,0.0007,price,8,20, ['rsi','log_high','log_low','log_close', 'ema_long', 'ema_short'])
        print(label)


    df = pd.DataFrame()
    df['timestep'] = time_step
    df['TRD_Outcome'] = label

    #Show
    print(df['TRD_Outcome'].value_counts())


    if type(label[0]) == int:
        class_2 = df[df['TRD_Outcome'] == 2] #losing trades
        class_2_sampled = class_2.sample(frac=0.5, random_state=42)# Randomly drop some class 2 instances
        df = pd.concat([df[df['TRD_Outcome'] != 2], class_2_sampled])

    else:
        df['TRD_Outcome'].replace(to_replace=['Loss','sell_loss','buy_loss','Sell_win'],value=0, inplace=True)
        df['TRD_Outcome'].replace(to_replace=['Buy_win'], value=1, inplace=True)
        class_2 = df[df['TRD_Outcome'] == 0 ] #losing trades
        class_2_sampled = class_2.sample(frac=0.5, random_state=42)# Randomly drop some class 2 instances
        df = pd.concat([df[df['TRD_Outcome'] != 'loss'], class_2_sampled])

    print(df['TRD_Outcome'].value_counts())

    np.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/tp_10_sl_10_timestep.npy", np.array(df.timestep))
    np.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/tp_10_sl_10_label.npy", df.TRD_Outcome)
    # np.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/timestep2.npy", allow_pickle=True)
    # np.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/label2.npy", allow_pickle=True)


#saving the clusters( as list )
time_step = np.load("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/tp_10_sl_10_timestep.npy", allow_pickle=True)
label = np.load("C:/Users/muyu2/OneDrive/Documents/DeepLearning/eda/tp_10_sl_10_label.npy", allow_pickle=True)

print(len(label))

X_raw = (np.concatenate(time_step)).reshape(len(time_step),21,6)
print("k meanss has began")
from tslearn.clustering import TimeSeriesKMeans
dtw_kmeans = TimeSeriesKMeans (n_clusters=30,tol=1e-3, random_state=42, n_jobs=-1, verbose=1,max_iter=200)
clusters = dtw_kmeans.fit_predict(X_raw)

#saving the clusters( as list )
with open('bbands','wb') as file:
    pickle.dump(clusters, file)

print('Saving complete')


#Looking for similarities
x = time_step.flatten().reshape(len(time_step), 4*15) #feature num by lookback period
inertia =[]

#K-Means clustering
for k in range(2,20):
    kmeans = KMeans(n_clusters=k, random_state=42, verbose=1)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)
    print(k)

#finding the optimal k
plt.plot(range(2,20), inertia)
plt.xlabel("Num of k clusters")
plt.ylabel("Inertia")
plt.show()









