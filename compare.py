import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from model import RentalNet
from prepare import make_loaders
from train import MAIN_RUN, REPORTS, SEED, run_epoch, train_model
from train_keras import load_arrays, train_keras_model

SEEDS = [SEED, SEED + 1, SEED + 2]


def run_pytorch(seed, config=MAIN_RUN):
    """Train the PyTorch main run on one seed. The split stays the same."""
    torch.manual_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        batch_size=config["batch_size"]
    )
    train_loader.generator.manual_seed(seed)
    model = RentalNet(hidden=config["hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start = time.perf_counter()
    history, best_epoch = train_model(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, config["epochs"]
    )
    seconds = time.perf_counter() - start

    _, test_mae = run_epoch(model, test_loader, nn.MSELoss())
    return {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": history[best_epoch - 1]["val_mae"],
        "test_mae": test_mae,
        "seconds": seconds,
    }


def run_keras(seed, data, config=MAIN_RUN):
    """Train the Keras main run on one seed, on the same prepared rows."""
    start = time.perf_counter()
    model, history, best_epoch = train_keras_model(config, data, seed=seed)
    seconds = time.perf_counter() - start

    x_test, y_test = data[2]
    predictions = model.predict(x_test, verbose=0).flatten()
    test_mae = float(np.mean(np.abs(predictions - y_test)))
    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": float(history.loc[best_epoch - 1, "val_mae"]),
        "test_mae": test_mae,
        "seconds": seconds,
    }


def main():
    """Train both frameworks on every seed and write reports/comparison.csv."""
    REPORTS.mkdir(exist_ok=True)
    data = load_arrays()
    rows = []
    for seed in SEEDS:
        rows.append(run_pytorch(seed))
        rows.append(run_keras(seed, data))
    table = pd.DataFrame(rows).round(3)
    table.to_csv(REPORTS / "comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
