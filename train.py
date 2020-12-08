"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from GRU import get_gru
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/GRU.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/GRU_loss.csv', encoding='utf-8', index=False)

    # draw the loss curve of the training set and validation set
    import matplotlib.pyplot as plt
    
    history_dict = hist.history
    history_dict.keys()

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b',color = 'blue', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b',color='red', label='Validation loss')
    plt.rc('font', size = 18)
    plt.title('Training and validation loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def main():
    
    lag = 4
    config = {"batch": 256, "epochs": 600}
    raw_data = 'data/Radar_Traffic_Counts.csv'
    X_train, y_train, _, _, _ = process_data(raw_data, lag)
    # 2 layer - gru
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    m = get_gru([4, 64, 64, 1])
    train_model(m, X_train, y_train, config)

if __name__ == '__main__':
    main()
