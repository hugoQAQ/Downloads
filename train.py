import numpy as np
import pandas as pd
from data.data import process_data
from evaluation import MAPE, eva_regress, plot_results
from model.model import get_lstm, get_gru
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


def train_model(model, X_train, y_train, name,config):
    """train
    train a NN model.

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

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)

    history_dict = hist.history
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

    plt.savefig('images/'+name+'Training/validation loss plot')
