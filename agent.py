from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras import Model


class Agent(object):
    def __init__(self, name='model', input_num=None, output_num=None):
        """A learning agent that uses tensorflow to create a neural network"""
        assert input_num is not None
        assert output_num is not None
        self.input_num = input_num
        self.output_num = output_num
        self._build_net()

    def _build_net(self):
        """Construct the neural network"""

        # Change the network structure here
        S = Input(shape=[self.input_num])
        h0 = Dense(300, activation="sigmoid")(S)
        h1 = Dense(600, activation="sigmoid")(h0)
        h2 = Dense(29, activation="sigmoid")(h1)
        V = Dense(self.output_num, activation="sigmoid")(h2)
        self.model = Model(inputs=S, outputs=V)
        self.model.compile(optimizer="adam", loss='mse')

    def train(self, x, y, n_epoch=100, batch=32):
        """Train the network"""
        self.model.fit(x=x, y=y, epochs=n_epoch, batch_size=batch)

    def predict(self, x):
        """Input values to the neural network and return the result"""
        a = self.model.predict(x)
        return a
