import numpy as np
import torch

from bikes_data import make_rentals
from prepare import (
    NUMERIC_COLUMNS,
    BikeDataset,
    make_loaders,
    prepare_features,
    split_frame,
)


def test_dataset_returns_float32_tensors():
    features = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    targets = np.arange(10, dtype=np.float32)
    dataset = BikeDataset(features, targets)

    assert len(dataset) == 10
    x, y = dataset[3]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (8,)
    assert y.shape == ()


def test_split_sizes_add_up(frame, splits):
    train, val, test = splits
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(train) + len(val) + len(test) == len(frame)


def test_splits_do_not_overlap():
    tagged = make_rentals().assign(row_id=range(6000))
    train, val, test = split_frame(tagged)
    ids = [set(part["row_id"]) for part in (train, val, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_scaler_is_fitted_on_training_rows_only(splits):
    (x_train, _), (x_val, _), _ = prepare_features(*splits)
    n = len(NUMERIC_COLUMNS)
    assert np.allclose(x_train[:, :n].mean(axis=0), 0, atol=1e-4)
    assert not np.allclose(x_val[:, :n].mean(axis=0), 0, atol=1e-3)


def test_train_loader_shuffles_but_is_repeatable():
    first, _, _ = make_loaders(seed=7)
    second, _, _ = make_loaders(seed=7)
    _, y_first = next(iter(first))
    _, y_second = next(iter(second))
    assert torch.equal(y_first, y_second)

    unshuffled = first.dataset.targets[: len(y_first)]
    assert not torch.equal(y_first, unshuffled)


def test_validation_loader_does_not_shuffle():
    _, val_loader, _ = make_loaders()
    _, y_batch = next(iter(val_loader))
    assert torch.equal(y_batch, val_loader.dataset.targets[: len(y_batch)])
