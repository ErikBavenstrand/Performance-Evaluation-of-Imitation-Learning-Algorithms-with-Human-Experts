from tensorflow.python.keras import layers, models


class Agent(object):
    def __init__(self, name='model', input_num=None, output_num=None):
        """Initialize the agent"""
        assert input_num is not None
        assert output_num is not None
        self.model = models.Sequential()
        self.input_num = input_num
        self.output_num = output_num
        self._build_net()

    def _build_net(self):
        """Construct the neural network"""
        self.model.add(layers.Dense(32, input_dim=self.input_num,
                                    activation='linear'))
        self.model.add(layers.Dense(16, activation='linear'))
        self.model.add(layers.Dense(self.output_num, activation='linear'))
        self.model.compile(optimizer='rmsprop', loss='mse')

    def train(self, x, y, n_epoch=100, batch=32):
        """Train the network"""
        self.model.fit(x=x, y=y, epochs=n_epoch, batch_size=batch)

    def predict(self, x):
        """Input values to the neural network and return the result"""
        a = self.model.predict(x)
        return a
