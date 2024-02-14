from keras import layers
from keras.layers import Dense

def model_LSTM(inp):
#     x = Lambda(lambda x: x[:, slice_indx])(inp)
    x = layers.LSTM(units=50, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=50, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=50, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units=1024)(x)
    x = layers.Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    return x