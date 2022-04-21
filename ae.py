# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


class DataGen:
    def __init__(self, A_range, w_range, o_range, b_range, e_range=(0, 0), tics=100):
        """
        mega docstring
        """
        self.tics = tics

        limits = [A_range, w_range, o_range, b_range, e_range]
        self.LOW = [limit[0] for limit in limits]
        self.UP = [limit[1] for limit in limits]

    def get_data(self, n=1, get_params=False):
        """
        cute docstring
        """
        size = (n, 5)
        params = np.random.uniform(self.LOW, self.UP, size=size)
        A, w, o, b, e = [p.reshape(-1, 1) for p in params.T]
        t = np.linspace(0, 10, self.tics)

        u = A * np.sin(w * t) * np.exp(-o * t)
        u_noise = u + np.random.normal(0, e, size=(n, self.tics))

        if get_params:
            return u_noise, u, [A, w, o, b, e]
        else:
            return u_noise, u


class DAE(keras.Model):
    def __init__(self, encoder, decoder, intervaler, latent_dim):
        super(DAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.intervaler = intervaler
        self.latent_dim = latent_dim

    def compile(self, optimizer, loss_fn):
        super(DAE, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, x, encoded_data=False, interval=False):
        encoded = self.encoder(x)
        if encoded_data:
            return encoded
        elif interval:
            return self.intervaler(encoded)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            predictions = self(x)
            loss = self.loss_fn(predictions, y)
        grads = tape.gradient(
            loss, self.encoder.trainable_weights + self.decoder.trainable_weights
        )
        self.optimizer.apply_gradients(
            zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights)
        )

        noise_abs = tf.abs(self(x) - y)

        with tf.GradientTape() as tape:
            predictions = intervaler(self(x, encoded_data=True))
            loss = self.loss_fn(predictions, noise_abs)
        grads = tape.gradient(loss, self.intervaler.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.intervaler.trainable_weights))

    def train(self, X, Y, epochs):
        for x, y in tqdm(zip(X, Y)):
            data = [np.atleast_2d(x), np.atleast_2d(y)]
            for epoch in range(epochs):
                self.train_step(data)


latent_dim = 4
tics = 100

DG = DataGen(
    A_range=(0.1, 1),
    w_range=(0.1, 1),
    o_range=(0.1, 1),
    b_range=(0.1, 1),
    e_range=(0, 0.02),
    tics=tics,
)

encoder = keras.Sequential(
    [
        keras.Input(shape=(tics,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(latent_dim),
    ],
    name="encoder",
)
decoder = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(tics),
    ],
    name="decoder",
)
intervaler = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(tics),
    ],
    name="decoder",
)

dae = DAE(
    encoder=encoder, decoder=decoder, intervaler=intervaler, latent_dim=latent_dim,
)
dae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.003),
    loss_fn=keras.losses.MeanSquaredError(),
)

x, y = DG.get_data(n=1000, get_params=False)
dae.train(x, y, epochs=100)

fig, ax = plt.subplots(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(x[0])
plt.plot(y[0])
plt.subplot(1, 2, 2)
plt.plot(dae.predict(x)[0])
plt.plot(dae(x, interval=True)[0])

plt.plot(y[0])
plt.plot(dae.predict(x)[0])

x_test, y_test = DG.get_data(get_params=False)
fig, ax = plt.subplots(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(x_test[0])
plt.plot(y_test[0])
plt.subplot(1, 2, 2)
plt.plot(dae.predict(x_test)[0])
plt.plot(dae.predict(x_test)[0] + dae(x_test, interval=True)[0], "y")
plt.plot(dae.predict(x_test)[0] - dae(x_test, interval=True)[0], "y")

plt.plot(y_test[0])
plt.plot(dae.predict(x_test)[0])

with open("params", "w+") as file:
    for param in [encoder, decoder, intervaler]:
        for weight in param.trainable_weights:
            file.write(str(weight))
        file.write("\n\n")
