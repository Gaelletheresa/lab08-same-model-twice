import keras
import pandas as pd

from keras_model import build_keras_model
from prepare import make_loaders
from train import MAIN_RUN, REPORTS, SEED, arrays_from, plot_curves

def load_arrays(batch_size=MAIN_RUN["batch_size"]):
    """The same prepared rows PyTorch trains on, as NumPy arrays."""
    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size)
    return (
        arrays_from(train_loader),
        arrays_from(val_loader),
        arrays_from(test_loader),
    )


def train_keras_model(config, data, checkpoint_path=None, seed=SEED):
    """Train the Keras network with compile and fit.

    Returns (model, history, best_epoch). The history has one row per epoch
    with train_loss, val_loss and val_mae. Epochs are numbered from 1.
    """
    keras.utils.set_random_seed(seed)   
    (x_train, y_train), (x_val, y_val), _ = data

    model = build_keras_model(hidden=config["hidden"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="mse",
        metrics=["mae"],
    )

    # patience equal to the number of epochs: the run never stops early,
    # so it trains for as long as the PyTorch run does.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["epochs"],
            restore_best_weights=True,
        )
    ]
    if checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )

    fitted = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=0,
    )

    history = pd.DataFrame(
        {
            "epoch": range(1, len(fitted.history["loss"]) + 1),
            "train_loss": fitted.history["loss"],
            "val_loss": fitted.history["val_loss"],
            "val_mae": fitted.history["val_mae"],
        }
    )
    best_epoch = int(history["val_loss"].idxmin()) + 1
    return model, history, best_epoch


def main():
    """Train the Keras main run and save its history, plot and best model."""
    REPORTS.mkdir(exist_ok=True)
    data = load_arrays()
    model, history, best_epoch = train_keras_model(
        MAIN_RUN, data, REPORTS / "best_model_keras.keras"
    )
    history.to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(
        history.to_dict("records"),
        best_epoch,
        REPORTS / "learning_curves_keras.png",
        "Keras main run",
    )
    best = history.loc[best_epoch - 1]
    print(f"best epoch {best_epoch}, validation MAE {best['val_mae']:.3f}")


if __name__ == "__main__":
    main()
