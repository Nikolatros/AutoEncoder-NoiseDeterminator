import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from numpy.random.mtrand import RandomState


class DataGen:
    def __init__(
        self, A_range, w_range, o_range, b_range, e_range=(0, 0), n=1, tics=100
    ):
        """
        mega docstring
        """
        self.tics = tics
        self.size = (n, 5)

        limits = [A_range, w_range, o_range, b_range, e_range]
        self.LOW = [limit[0] for limit in limits]
        self.UP = [limit[1] for limit in limits]

    def get_data(self, get_params=False):
        """
        cute docstring
        """
        params = np.random.uniform(self.LOW, self.UP, size=self.size)
        A, w, o, b, e = [p.reshape(-1, 1) for p in params.T]
        t = np.linspace(0, 10, self.tics)

        u = A * np.sin(w * t) * np.exp(-o * t)
        u_noise = u + np.random.normal(0, e, size=(1, self.tics))

        u = np.concatenate([u.reshape(self.tics, 1), t.reshape(-1, 1)], axis=1)
        u_noise = np.concatenate(
            [u_noise.reshape(self.tics, 1), t.reshape(-1, 1)], axis=1
        )

        if get_params:
            return u_noise, u, [A, w, o, b, e]
        else:
            return u_noise, u


class DAE(keras.Model):
    def __init__(self, encoder, decoder, decoder_second, latent_dim):
        super(DAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_second = decoder_second
        self.latent_dim = latent_dim

    def compile(self, e_optimizer, d_optimizer, loss_fn):
        super(DAE, self).compile()
        self.e_optimizer = e_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        print("start")
        x, y = data

        with tf.GradientTape() as tape:
            lattent_space = self.encoder(x)
            predictions = self.decoder(lattent_space)
            d_loss = self.loss_fn(y, predictions)
        grads = tape.gradient(
            d_loss, self.encoder.trainable_weights + self.decoder.trainable_weights
        )
        self.d_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

        err_abs = tf.abs(predictions - y)

        with tf.GradientTape() as tape:
            predictions = self.decoder_second(
                self.encoder(x)
            )  # получается, что второй декодер "на шаг впереди"
            d_loss = self.loss_fn(err_abs, predictions)
        grads = tape.gradient(d_loss, self.decoder_second.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.decoder_second.trainable_weights)
        )
        print("end")


n = 1
latent_dim = 1

DG = DataGen(
    A_range=(0.1, 1),
    w_range=(0.1, 1),
    o_range=(0.1, 1),
    b_range=(0.1, 1),
    e_range=(0, 0),
    n=n,
    tics=100,
)

x, y = DG.get_data(get_params=False)

encoder = keras.Sequential(
    [keras.Input(shape=(2,)), layers.LeakyReLU(alpha=0.2), layers.Dense(latent_dim),],
    name="encoder",
)

decoder = keras.Sequential(
    [keras.Input(shape=(latent_dim,)), layers.LeakyReLU(alpha=0.2), layers.Dense(2),],
    name="decoder",
)

decoder_second = keras.Sequential(
    [keras.Input(shape=(latent_dim,)), layers.LeakyReLU(alpha=0.2), layers.Dense(2),],
    name="decoder",
)


dae = DAE(
    encoder=encoder,
    decoder=decoder,
    decoder_second=decoder_second,
    latent_dim=latent_dim,
)
dae.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    e_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.MeanSquaredError(),
)

dae.fit(x, y)
