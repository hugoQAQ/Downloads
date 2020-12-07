import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers
from keras import layers
from keras.layers import LSTM,GRU

# Data processing
df = pd.read_csv("/kaggle/input/radar-traffic-data/Radar_Traffic_Counts.csv")
df=df.dropna()
df_copy=df.copy()
df['datetime']=pd.to_datetime(df_copy[['Year','Month','Day','Hour','Minute']])
df_blk=df[df['location_name']=='3201 BLK S LAMAR BLVD (BROKEN SPOKE)'].sort_values(by='datetime').copy()
# Training set
df_blk_1908=df_blk[(df_blk['Year']==2019) & (df_blk['Month']==8)]

grb_datetime_flow_sum=df_blk_1908.groupby('datetime')['Volume'].sum()
train=pd.DataFrame({'datetime':grb_datetime_flow_sum.index,'volume':grb_datetime_flow_sum})

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(train['volume'].values.reshape(-1, 1))
result=scaler.transform(train['volume'].values.reshape(-1, 1)).reshape(1, -1)[0]

tr=[]
for i in range(1, len(result)):
    tr.append(result[i - 1: i + 1])
tr = np.array(tr)
np.random.shuffle(tr)
X_train = tr[:, :-1]
y_train = tr[:, -1]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#Validation set
df_blk_1909=df_blk[(df_blk['Year']==2019) & (df_blk['Month']==9)]

grb_datetime_flow_sum=df_blk_1909.groupby('datetime')['Volume'].sum()
validation=pd.DataFrame({'datetime':grb_datetime_flow_sum.index,'volume':grb_datetime_flow_sum})

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(validation['volume'].values.reshape(-1, 1))
result=scaler.transform(validation['volume'].values.reshape(-1, 1)).reshape(1, -1)[0]

val=[]
for i in range(1, len(result)):
    val.append(result[i - 1: i + 1])
val = np.array(val)
np.random.shuffle(val)
X_val = val[:, :-1]
y_val = val[:, -1]
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

#Model
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM,GRU
from keras.layers.core import Dense, Dropout, Activation, Flatten

model = Sequential()
model.add(LSTM(128, input_shape=(1, 1), return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
#Training
LSTM_model = model
LSTM_model.compile(loss='mean_squared_error', optimizer=Adam())

from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
history = LSTM_model.fit(X_train, y_train, epochs=200, batch_size=200, verbose=2, shuffle=False, validation_data=(X_val, y_val))
