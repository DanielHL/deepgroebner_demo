import numpy as np
import tensorflow as tf

class ParallelMultilayerPerceptron(tf.keras.Model):
    """A parallel multilayer perceptron network.

    This model expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the hidden layers.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

    Examples
    --------
    >>> pmlp = ParallelMultilayerPerceptron([128])
    >>> states = tf.constant([
            [[ 0,  1],
             [ 3,  0],
             [-1, -1]],
            [[ 8,  5],
             [ 3,  3],
             [ 3,  5]],
            [[ 6,  7],
             [ 6,  8],
             [-1, -1]],
        ])
    >>> logprobs = pmlp(states)
    >>> logprobs.shape
    TensorShape([3, 3])
    >>> actions = tf.random.categorical(logprobs, 1)
    >>> actions.shape
    TensorShape([3, 1])

    """

    def __init__(self, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(ParallelMultilayerPerceptron, self).__init__()
        self.embedding = ParallelEmbeddingLayer(hidden_layers[-1], hidden_layers[:-1],
                                                activation=activation, final_activation=activation)
        self.deciding = ParallelDecidingLayer([], final_activation=final_activation)

    def call(self, batch):
        X = self.embedding(batch)
        X = self.deciding(X)
        return X


class ParallelDecidingLayer(tf.keras.layers.Layer):
    """A layer for computing probability distributions over arbitrary numbers of rows.

    This layer is used following an embedding and processing of the state of a LeadMonomialsWrapper
    to produce the policy probabilities for each available action. The layer learns a single function
    implemented with a multilayer perceptron that computes scores independently for each row. These
    scores are softmaxed to produce the probabilites.

    Parameters
    ----------
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the hidden layers.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

    """

    def __init__(self, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(ParallelDecidingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(1, activation='linear')
        self.final_activation = tf.nn.log_softmax if final_activation == 'log_softmax' else tf.nn.softmax

    def call(self, batch, mask=None):
        """Return probability distributions over rows of batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, feature_dim)
            Batch of feature vectors with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim)
            Softmaxed probability distributions over valid rows.

        """
        X = batch
        for layer in self.hidden_layers:
            X = layer(X)
        X = tf.squeeze(self.final_layer(X), axis=-1)
        if mask is not None:
            X += tf.cast(~mask, tf.float32) * -1e9
        output = self.final_activation(X)
        return output


class ParallelEmbeddingLayer(tf.keras.layers.Layer):
    """A layer for computing a nonlinear embedding of non-negative integer feature vectors.

    This layer is used with the LeadMonomialsEnv to embed the exponent vectors of pairs
    into feature vectors. Each vector is embedded independently by a single learned multilayer
    perceptron. A mask is generated and attached to the output based on padding by -1.

    Parameters
    ----------
    embed_dim : int
        Positive integer output dimension of the embedding.
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation used for the hidden layers.
    final_activation : {'relu', 'linear', 'exponential'}, optional
        Activation used for the final output embedding layer.

    """

    def __init__(self, embed_dim, hidden_layers, activation='relu', final_activation='relu'):
        super(ParallelEmbeddingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(embed_dim, activation=final_activation)

    def call(self, batch):
        """Return the embedding for this batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.int32` and shape (batch_dim, padded_dim, input_dim)
            Input batch, with padded rows indicated by -1 and all other values non-negative.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, embed_dim)
            Embedding of the input batch with attached mask indicating valid rows.

        """
        X = tf.cast(batch, tf.float32)
        for layer in self.hidden_layers:
            X = layer(X)
        output = self.final_layer(X)
        return output

    def compute_mask(self, batch, mask=None):
        return tf.math.not_equal(batch[:, :, -1], -1)


class PairsLeftBaseline:
    """A Buchberger value network that returns discounted pairs left."""

    def __init__(self, gam=0.99):
        self.gam = gam
        self.trainable_variables = []

    def predict(self, X, **kwargs):
        states, pairs, *_ = X.shape
        if self.gam == 1:
            fill_value = - pairs
        else:
            fill_value = - (1 - self.gam ** pairs) / (1 - self.gam)
        return np.full((states, 1), fill_value)

    def __call__(self, inputs):
        return self.predict(inputs)

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass


class AgentBaseline:
    """A Buchberger value network that returns an agent's performance."""

    def __init__(self, agent, gam=0.99):
        self.agent = agent
        self.gam = gam
        self.trainable_variables = []

    def predict(self, env):
        env = env.copy()
        R = 0.0
        discount = 1.0
        state = (env.G, env.P) if hasattr(env, 'G') else env._matrix()
        done = False
        while not done:
            action = self.agent.act(state)
            state, reward, done, _ = env.step(action)
            R += reward * discount
            discount *= self.gam
        return R

    def __call__(self, inputs):
        return self.predict(inputs)

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass
