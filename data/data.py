"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(raw_data,location,lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        raw_data: String, name of .csv raw data file.
        location: String, name of street.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """

    # Data processing
    df = pd.read_csv(raw_data, encoding='utf-8').fillna(0)
    df['datetime']=pd.to_datetime(df[['Year','Month','Day','Hour','Minute']].copy())
    df['location_name']=df['location_name'].str.lstrip()
    df_street=df[(df['location_name']==location)].sort_values(by='datetime').copy()

    # Training set
    df_street_1836=df_street[(df_street['Year']==2018)&(df_street['Month']<7)].copy()

    # sum of volume by time bin
    grb_datetime_flow_sum=df_street_1836.groupby('datetime')['Volume'].sum()
    train=pd.DataFrame({'datetime':grb_datetime_flow_sum.index,'volume':grb_datetime_flow_sum})

    # Normalization of training set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train['volume'].values.reshape(-1, 1))
    result=scaler.transform(train['volume'].values.reshape(-1, 1)).reshape(1, -1)[0]

    train=[]
    for i in range(lags, len(result)):
        train.append(result[i - lags: i + 1])
    train = np.array(train)
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Test set
    df_street_1807=df_street[(df_street['Year']==2018) & (df_street['Month']==7)]
    grb_datetime_flow_sum=df_street_1807.groupby('datetime')['Volume'].sum()
    validation=pd.DataFrame({'datetime':grb_datetime_flow_sum.index,'volume':grb_datetime_flow_sum})

    # Normalization of test set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(validation['volume'].values.reshape(-1, 1))
    result=scaler.transform(validation['volume'].values.reshape(-1, 1)).reshape(1, -1)[0]

    test=[]
    for i in range(lags, len(result)):
        test.append(result[i - lags: i + 1])
    test = np.array(test)
    np.random.shuffle(test)
    X_test = test[:, :-1]
    y_test = test[:, -1]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    return X_train, y_train, X_test, y_test, scaler
