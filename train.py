import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from model import RentalNet
from prepare import make_loaders

REPORTS = Path("reports")
SEED = 2026

MAIN_RUN = {
    "name": "main",
    "hidden": 64,
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 200,
}
EXPERIMENTS = [
    {**MAIN_RUN, "name": "lr_too_small", "lr": 0.00001},
    {**MAIN_RUN, "name": "lr_too_large", "lr": 0.1},
    {**MAIN_RUN, "name": "wide_network", "hidden": 512},
]
FINAL_RUN = "wide_network"


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Run one epoch. Trains if given an optimizer, otherwise only measures.

    Returns the mean loss and the mean absolute error (in rentals).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_abs_error = 0.0
    total_rows = 0

    with torch.set_grad_enabled(training):
        for features, targets in loader:
            predictions = model(features)
            loss = loss_fn(predictions, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            rows = len(targets)
            total_loss += loss.item() * rows
            total_abs_error += (predictions - targets).abs().sum().item()
            total_rows += rows

    return total_loss / total_rows, total_abs_error / total_rows


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    """Train for many epochs, log every epoch, restore the best weights.

    Returns (history, best_epoch). The epochs are numbered from 1.
    """
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return history, best_epoch


def arrays_from(loader):
    """Get the prepared features and targets back out of a loader as arrays."""
    return loader.dataset.features.numpy(), loader.dataset.targets.numpy()


def baseline_metrics(train_loader, val_loader, test_loader):
    """Score the training-mean baseline and linear regression (MAE and MSE)."""
    x_train, y_train = arrays_from(train_loader)
    mean_value = float(y_train.mean())
    linear = LinearRegression().fit(x_train, y_train)

    results = {"train_mean": {}, "linear": {}}
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        x, y = arrays_from(loader)
        predictions = {
            "train_mean": np.full_like(y, mean_value),
            "linear": linear.predict(x),
        }
        for name, predicted in predictions.items():
            results[name][f"{split}_mae"] = float(mean_absolute_error(y, predicted))
            results[name][f"{split}_mse"] = float(mean_squared_error(y, predicted))
    return results


def run_experiment(config):
    """Train one network from the settings in `config`. Seeded, so repeatable."""
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = make_loaders(batch_size=config["batch_size"])
    model = RentalNet(hidden=config["hidden"])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, config["epochs"]
    )
    return model, history, best_epoch


def summarize(config, history, best_epoch):
    """One row of the experiments table."""
    best = history[best_epoch - 1]
    last = history[-1]
    return {
        "run": config["name"],
        "hidden": config["hidden"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_mae": round(best["val_mae"], 3),
        "final_train_loss": round(last["train_loss"], 3),
        "final_val_loss": round(last["val_loss"], 3),
    }


def plot_curves(history, best_epoch, path, title):
    """Plot training and validation loss for every epoch, and mark the best."""
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["epoch"], frame["train_loss"], label="training loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax.axvline(
        best_epoch, color="gray", linestyle="--", label=f"best epoch {best_epoch}"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, log scale)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders()

    baselines = baseline_metrics(train_loader, val_loader, test_loader)
    for name, scores in baselines.items():
        print(
            f"{name:11s} val MAE {scores['val_mae']:6.3f}  "
            f"val MSE {scores['val_mse']:8.3f}"
        )

    rows = []
    models = {}
    best_rows = {}
    for config in [MAIN_RUN, *EXPERIMENTS]:
        model, history, best_epoch = run_experiment(config)
        models[config["name"]] = model
        best_rows[config["name"]] = history[best_epoch - 1]
        summary = summarize(config, history, best_epoch)
        rows.append(summary)
        print(
            f"{summary['run']:13s} best epoch {summary['best_epoch']:3d}  "
            f"best val MAE {summary['best_val_mae']:7.3f}  "
            f"final train {summary['final_train_loss']:9.3f}  "
            f"final val {summary['final_val_loss']:9.3f}"
        )

        if config["name"] == "main":
            pd.DataFrame(history).to_csv(REPORTS / "history.csv", index=False)
            curve_path = REPORTS / "learning_curves.png"
        else:
            curve_path = REPORTS / f"curves_{config['name']}.png"
        plot_curves(history, best_epoch, curve_path, f"Run: {config['name']}")

    pd.DataFrame(rows).to_csv(REPORTS / "experiments.csv", index=False)

    # The test rows are used once, for the network chosen on the validation rows.
    test_mse, test_mae = run_epoch(models[FINAL_RUN], test_loader, nn.MSELoss())
    metrics = {
        "baselines": baselines,
        "network": {
            "run": FINAL_RUN,
            "val_mae": best_rows[FINAL_RUN]["val_mae"],
            "val_mse": best_rows[FINAL_RUN]["val_loss"],
            "test_mae": test_mae,
            "test_mse": test_mse,
        },
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"final network ({FINAL_RUN})  test MAE {test_mae:.3f}  test MSE {test_mse:.3f}"
    )
