from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import set_learning_phase
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import SGD
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
            h0 = Dense(256, activation="relu")(S)
            h1 = Dense(128, activation="relu")(h0)
            V = Dense(self.output_num, activation="sigmoid")(h1)
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

    def _activation_function(self, arr, sigmoid=False):
        """Activation function for array"""
        if sigmoid:
            arr = self._sigmoid(arr)
        else:
            arr = self._relu(arr)
        return arr

    def _sigmoid(self, x):
        """Sigmoid function for a single value"""
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def train(self, x, y, n_epoch=10000, batch=32):
        """Train the network"""
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        for i in range(len(self.models)):
            mc = ModelCheckpoint('best_model_' + str(i) + '.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            self.models[i].fit(x=x, y=y, epochs=n_epoch, batch_size=batch, validation_split=0.1, verbose=1, callbacks=[es, mc])
        for i in range(len(self.models)):
            self.models[i] = load_model('best_model_' + str(i) + '.h5')
        self._store_weights()

    def predict(self, x):
        """Input values to the neural network and return the result"""
        result = np.zeros((0, self.output_num))
        for i in range(len(self.models)):
            z = x
            for j in range(len(self.weights[i])):
                z = np.matmul(z, self.weights[i][j]) + self.biases[i][j]
                if len(self.weights[i]) - j == 1:
                    z = self._activation_function(z, sigmoid=True)
                else:
                    z = self._activation_function(z)
            result = np.concatenate((result, z))
        return result

    def predict_slow(self, x):
        return self.models[0].predict(x)

    def mean_of_prediction(self, prediction):
        """Computes the mean of a prediction"""
        result = []
        for j in range(len(prediction[0])):
            total_sum = 0
            for i in range(len(prediction)):
                total_sum += prediction[i][j]
            result.append(total_sum / len(prediction))
        return [result]

    def covariance_of_prediction(self, prediction, mean):
        cov = np.zeros(shape=prediction.shape)
        for i in range(len(prediction)):
            cov[i] = prediction[i] - mean
        cov_t = np.transpose(cov)
        cov = np.dot(cov_t, cov)
        diag = np.zeros(shape=[len(cov)])
        for i in range(len(cov)):
            diag[i] = cov[i][i]
        result = np.linalg.norm(diag)
        return result

    def save(self, folder, name):
        for m in range(len(self.models)):
            self.models[m].save(folder + str(m) + "_" + name + ".h5")
            print("Model %d saved" % m)

    def load(self, folder, name):
        for m in range(len(self.models)):
            print("Model: %d" % m)
            self.models[m] = load_model(folder + str(m) + "_" + name + ".h5")
        self._store_weights()