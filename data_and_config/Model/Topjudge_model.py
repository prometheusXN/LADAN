import tensorflow as tf
from keras.models import Model
import keras
from Model.CNN_cell import CNNEncoder
from Model.LSTM_cell import LSTMDecoder


class Topjudge(Model):
    def __init__(self, config, num1, num2):
        super(Topjudge, self).__init__()

        self.config = config
        self.encoder = CNNEncoder(self.config)
        self.decoder = LSTMDecoder(self.config, num1, num2)
        self.trans_linear = keras.layers.Dense(self.decoder.feature_len)
        self.dropout = keras.layers.Dropout(self.config.getfloat('train', 'dropout'))
        self.num1 = num1
        self.num2 = num2

    def call(self, inputs, mask=None):
        x = self.encoder(inputs)
        if self.encoder.feature_len != self.decoder.feature_len:
            x = self.trans_linear(x)

        x = self.dropout(x)
        x = self.decoder(x)

        return x

    def compute_output_shape(self, input_shape):

        return [(input_shape[0], self.num1), (input_shape[0], self.num2), (input_shape[0], 12)]