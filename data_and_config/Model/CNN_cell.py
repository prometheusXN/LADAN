import tensorflow as tf
from keras.models import Model
import keras.layers as Layers
import keras

class CNNEncoder(Model):

    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        self.min_gram = config.getint("net", "min_gram")
        self.max_gram = config.getint("net", "max_gram")
        self.width = config.getint("data", "vec_size")
        self.filters = config.getint("net", "filters")
        self.batch_size = config.getint("data", "batch_size")


        self.convs = []
        for height in range(self.min_gram, self.max_gram + 1):
            self.convs.append(Layers.Conv2D(self.filters, [height, self.width], data_format='channels_first'))

        self.feature_len = (-self.min_gram + self.max_gram + 1) * self.filters

    def call(self, inputs, mask=None):
        conv_out = []
        inputs = tf.expand_dims(inputs, axis=1)

        for conv in self.convs:
            y = keras.activations.relu(tf.squeeze(conv(inputs)))
            y = tf.reduce_max(y, -1)
            conv_out.append(y)

        conv_out = tf.concat(conv_out, axis=1)

        return conv_out

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.feature_len)