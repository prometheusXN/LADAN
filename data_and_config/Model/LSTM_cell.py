import tensorflow as tf
from keras.models import Model
import keras.layers as Layers
import keras
from keras import backend as K
from keras.engine.base_layer import InputSpec

def get_num_classes(s, num1, num2):
    if s == "crit":
        return num2
        # return len(accusation_list)
    if s == "law":
        return num1
        # return len(law_list)
    if s == "time":
        return 12

def _standardize_args(inputs, initial_state, constants, num_constants):
    """Standardize `__call__` to a single list of tensor inputs.

    When running a model loaded from file, the input tensors
    `initial_state` and `constants` can be passed to `RNN.__call__` as part
    of `inputs` instead of by the dedicated keyword arguments. This method
    makes sure the arguments are separated and that `initial_state` and
    `constants` are lists of tensors (or None).

    # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None

    # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
    """
    if isinstance(inputs, list):
        assert initial_state is None and constants is None
        if num_constants is not None:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
        inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)

    return inputs, initial_state

class LSTMDecoder(Model):

    def __init__(self, config, num1, num2):
        super(LSTMDecoder, self).__init__()

        self.num1 = num1
        self.num2 = num2
        self.config = config
        self.feature_len = config.getint("net", "hidden_size")
        self.outfc = []
        self.task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
        for x in self.task_name:
            self.outfc.append(keras.layers.Dense(get_num_classes(x, num1, num2)))

        self.midfc = []
        for x in self.task_name:
            self.midfc.append(keras.layers.Dense(self.feature_len))

        self.graph = self.generate_graph(config)

        self.cell_list = []
        for x in range(len(self.task_name)+1):
            self.cell_list.append(keras.layers.LSTMCell(self.feature_len))

        self.hidden_state_fc_list = []
        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(keras.layers.Dense(self.feature_len))
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(keras.layers.Dense(self.feature_len))
            self.cell_state_fc_list.append(arr)

        self.hidden_list = []
        task_name = self.config.get("data", "type_of_label").replace(" ", "").split(",")
        for a in range(0, len(task_name) + 1):
            self.hidden_list.append([tf.zeros([self.config.getint("data", "batch_size"), self.feature_len]), tf.zeros([self.config.getint("data", "batch_size"), self.feature_len])])#h_0, c_0

    def generate_graph(self, config):
        s = config.get("data", "graph")
        arr = s.replace("[", "").replace("]", "").split(",")
        graph = []
        n = 0
        if (s == "[]"):
            arr = []
            n = 3
        for a in range(0, len(arr)):
            arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
            arr[a][0] = int(arr[a][0])
            arr[a][1] = int(arr[a][1])
            n = max(n, max(arr[a][0], arr[a][1]))

        n += 1
        for a in range(0, n):
            graph.append([])
            for b in range(0, n):
                graph[a].append(False)

        for a in range(0, len(arr)):
            graph[arr[a][0]][arr[a][1]] = True

        return graph

    def call(self, inputs, mask=None):
        outputs = []
        task_name = self.config.get("data", "type_of_label").replace(" ", "").split(",")
        inputs_shape = inputs.get_shape().as_list()
        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            x = self.cell_list[a]
            state = self.hidden_list[a]
            x.build(inputs_shape)
            _, (h, c) = x.call(inputs, state)
            # h, c = self.cell_list(fc_input, self.hidden_list[a])
            for b in range(1, len(task_name) + 1):
                if self.graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                        # hp, cp = self.hidden_state_fc_list[a][b](h), self.cell_state_fc_list[a][b](c)
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)
            # self.hidden_list[a] = h, c
            if self.config.getboolean("net", "more_fc"):
                outputs.append(
                    tf.reshape(self.outfc[a - 1](tf.nn.relu(self.midfc[a - 1](h))),[self.config.getint("data", "batch_size"), -1]))
            else:
                outputs.append(self.outfc[a - 1](h))
        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.num1), (input_shape[0], self.num2), (input_shape[0], 12)]
