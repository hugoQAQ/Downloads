import numpy as np
import pandas as pd
from data.data import process_data
from evaluation import MAPE, eva_regress, plot_results
from model.model import get_lstm, get_gru
from train import train_model
from tensorflow.keras.models import Model,load_model
import matplotlib.pyplot as plt



class predict_volume():
    def __init__(self,raw_data,location,units,lags,config):
        self.raw_data=raw_data
        self.location=location
        self.units=units
        self.lags=lags
        self.config=config
        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = process_data(self.raw_data,self.location,self.lags)

    def training(self):
        gru_model= get_gru(self.units)
        train_model(gru_model, self.X_train, self.y_train, 'GRU', self.config)
        lstm_model= get_lstm(self.units)
        train_model(lstm_model, self.X_train, self.y_train, 'LSTM', self.config)

    def model_evaluation(self):
        y_preds = []
        lstm = load_model('model/LSTM.h5')
        gru = load_model('model/GRU.h5')
        models = [lstm, gru]
        names = ['LSTM', 'GRU']
        for name, model in zip(names, models):
            file = 'images/' + name + '.png'
            # plot_model(model, to_file=file, show_shapes=True) # pydotplus.graphviz.InvocationException: GraphViz's executables not found
            predicted = model.predict(self.X_test)
            predicted = self.scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
            y_preds.append(predicted[:120])
            print(name)
            eva_regress(self.y_test, predicted)

        plot_results(self.y_test[: 120], y_preds, names)

def MAIN():
    lags = 4
    config = {"batch": 256, "epochs": 600}
    raw_data = 'data/Radar_Traffic_Counts.csv'
    location='100 BLK S CONGRESS AVE (Congress Bridge)'
    units=[lags,64,64,1]
    pv=predict_volume(raw_data,location,units,lags,config)
    pv.training()
    pv.model_evaluation()




if __name__ == '__main__':
    MAIN()
