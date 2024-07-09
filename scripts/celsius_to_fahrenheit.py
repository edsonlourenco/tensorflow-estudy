"""
Modelo de Machine Learning preditivo para conversão de Celsius em Fahrenheit!
O programa funciona por terminal onde o grau em Celsius é passado por parametro.

Fórmula: f = c x 1.8 + 32
"""

import argparse
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


def main():

    celsius_q    = np.array([40, -10,  0,  8, 15, 22,  38],   dtype=float)
    fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

    parser = argparse.ArgumentParser(description="Converte Fahrenhiet para Celsius.")
    parser.add_argument('celsius', type=float, help='Grau Celsius')
    
    args = parser.parse_args()
    
    celsius = args.celsius
    
    model = create_model()
    model = train_model(model, celsius_q, fahrenheit_a)
    prediction_fahrenheit = predict(model, celsius)

    print("O modelo prevê que {} graus Celsius é: {} graus Fahrenheit".format(celsius, prediction_fahrenheit))

if __name__ == '__main__':
    main()