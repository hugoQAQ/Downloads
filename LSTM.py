from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
#LSTM model
def get_LSTM(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: LSTM Model
    """
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model
