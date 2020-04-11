import tensorflow as tf
from keras.models import Model
import keras
from Model.CNN_cell import CNNEncoder
from Model.LSTM_cell import LSTMDecoder


class CNN_multi(Model):
    def __init__(self, config, num1, num2):
        super(CNN_multi, self).__init__()

        self.config = config
        self.encoder = CNNEncoder(self.config)
        self.feature_len = config.getint("net", "hidden_size")
        self.dropout = keras.layers.Dropout(self.config.getfloat('train', 'dropout'))
        self.num1 = num1
        self.num2 = num2
        self.decoder_1 = keras.layers.Dense(self.num1)
        self.decoder_2 = keras.layers.Dense(self.num2)
        self.decoder_3 = keras.layers.Dense(12)
        self.trans_linear = keras.layers.Dense(self.feature_len)

    def call(self, inputs, mask=None):
        x = self.encoder(inputs)
        if self.encoder.feature_len != self.feature_len:
            x = self.trans_linear(x)

        x = self.dropout(x)
        x = [self.decoder_1(x), self.decoder_2(x), self.decoder_3(x)]

        return x

    def compute_output_shape(self, input_shape):

        return [(input_shape[0], self.num1), (input_shape[0], self.num2), (input_shape[0], 12)]