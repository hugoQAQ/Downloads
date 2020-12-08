"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(raw_data, lags):
    """Process data
    Reshape and split train\val data.

    # Arguments
        raw_data: String, name of .csv raw data file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_val: ndarray.
        y_val: ndarray.
        scaler: StandardScaler.
    """

    # Data processing
    # df = pd.read_csv("/kaggle/input/radar-traffic-data/Radar_Traffic_Counts.csv")
    df = pd.read_csv(raw_data, encoding='utf-8').fillna(0)
    df=df.dropna()
    df_copy=df.copy()
    df['datetime']=pd.to_datetime(df_copy[['Year','Month','Day','Hour','Minute']])
    df_blk=df[df['location_name']=='3201 BLK S LAMAR BLVD (BROKEN SPOKE)'].sort_values(by='datetime').copy()
    
    # Training set
    df_blk_1836=df_blk[(df_blk['Year']==2018)&(df_blk['Month']<7)].copy()
    print(df_blk_1836['Time Bin'].unique())

    # sum of volume by time bin
    grb_datetime_flow_sum=df_blk_1836.groupby('datetime')['Volume'].sum()
    train=pd.DataFrame({'datetime':grb_datetime_flow_sum.index,'volume':grb_datetime_flow_sum})
    print(train.head())

    # Normalization of training set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train['volume'].values.reshape(-1, 1))
    result=scaler.transform(train['volume'].values.reshape(-1, 1)).reshape(1, -1)[0]
    print(result)

    train=[]
    for i in range(lags, len(result)):
        train.append(result[i - lags: i + 1])
    train = np.array(train)
    print(train)
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    df_blk_1807=df_blk[(df_blk['Year']==2018) & (df_blk['Month']==7)]
    grb_datetime_flow_sum=df_blk_1807.groupby('datetime')['Volume'].sum()
    validation=pd.DataFrame({'datetime':grb_datetime_flow_sum.index,'volume':grb_datetime_flow_sum})

    # Normalization of training set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(validation['volume'].values.reshape(-1, 1))
    result=scaler.transform(validation['volume'].values.reshape(-1, 1)).reshape(1, -1)[0]
    print(result)
    val=[]
    for i in range(lags, len(result)):
        val.append(result[i - lags: i + 1])
    val = np.array(val)
    np.random.shuffle(val)
    X_val = val[:, :-1]
    y_val = val[:, -1]
    # X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    return X_train, y_train, X_val, y_val, scaler
