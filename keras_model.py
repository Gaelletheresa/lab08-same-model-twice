import keras
from keras import layers

from model import N_FEATURES


def build_keras_model(n_features=N_FEATURES, hidden=64):
    """The Keras twin of RentalNet: two hidden ReLU layers, one linear output."""
    return keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(1),
        ]
    )
