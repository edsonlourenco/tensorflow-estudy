import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def create_model():
    l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
    l1 = tf.keras.layers.Dense(units=4)
    l2 = tf.keras.layers.Dense(units=1)

    model = tf.keras.Sequential([l0, l1, l2])
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.legacy.Adam(0.1))

    return model


def train_model(model, celsius_q, fahrenheit_a):
    model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
    return model


def predict(model, celsius):
    prediction_fahrenheit = model.predict([celsius])

    return prediction_fahrenheit