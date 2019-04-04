from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import set_learning_phase
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras import Model
import numpy as np


class Agent(object):
    def __init__(self, input_num=None, output_num=None, ensemble_count=1):
        """A learning agent that uses tensorflow to create a neural network"""
        assert input_num is not None
        assert output_num is not None
        assert ensemble_count > 0

        self.ensemble_count = ensemble_count
        self.input_num = input_num
        self.output_num = output_num
        self.models = []

        self._build_net()

    def _build_net(self):
        """Construct the neural network"""
        for i in range(self.ensemble_count):
            # Change the network structure here
            S = Input(shape=[self.input_num])
            h0 = Dense(29, activation="sigmoid")(S)
            h1 = Dense(29, activation="sigmoid")(h0)
            h2 = Dense(29, activation="sigmoid")(h1)
            V = Dense(self.output_num, activation="sigmoid")(h2)
            model = Model(inputs=S, outputs=V)
            model.compile(optimizer="adam", loss='mse')
            self.models.append(model)
        self.weights = []
        self.biases = []
        for i in range(len(self.models)):
            self.weights.append([])
            self.biases.append([])
            for j in range(len(self.models[i].layers) - 1):
                self.weights[i].append([])
                self.biases[i].append([])
        self._store_weights()

    def _store_weights(self):
        """Store weights and biases of networks for quick numpy predictions"""
        for i in range(len(self.models)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = \
                    self.models[i].layers[j + 1].get_weights()[0]
                self.biases[i][j] = \
                    self.models[i].layers[j + 1].get_weights()[1]

    def _activation_function(self, arr):
        """Activation function for array"""
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                arr[i][j] = self._sigmoid(arr[i][j])
        return arr

    def _sigmoid(self, x):
        """Sigmoid function for a single value"""
        return 1 / (1 + np.e ** -x)

    def train(self, x, y, n_epoch=100, batch=32):
        """Train the network"""
        for i in range(len(self.models)):
            self.models[i].fit(x=x, y=y, epochs=n_epoch, batch_size=batch)
        self._store_weights()

    def predict(self, x):
        """Input values to the neural network and return the result"""
        result = np.zeros((0, self.output_num))
        for i in range(len(self.models)):
            z = x
            for j in range(len(self.weights[i])):
                z = np.matmul(z, self.weights[i][j]) + self.biases[i][j]
                z = self._activation_function(z)
            result = np.concatenate((result, z))
        return result
