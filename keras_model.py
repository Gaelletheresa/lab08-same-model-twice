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


def copy_weights_to_keras(torch_model, keras_model):
    """Copy the weights of a RentalNet into its Keras twin.

    A PyTorch Linear weight is stored as (outputs, inputs), but a Keras
    Dense kernel is stored as (inputs, outputs), so the kernel is transposed.
    Biases are 1-D and are copied as they are.
    """
    linears = [layer for layer in torch_model.layers if hasattr(layer, "weight")]
    denses = [layer for layer in keras_model.layers if layer.weights]
    for linear, dense in zip(linears, denses, strict=True):
        kernel = linear.weight.detach().numpy().T
        bias = linear.bias.detach().numpy()
        dense.set_weights([kernel, bias])
