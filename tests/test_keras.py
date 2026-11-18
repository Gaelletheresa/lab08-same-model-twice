import keras
import numpy as np
import pytest
import torch

from keras_model import build_keras_model, copy_weights_to_keras
from model import N_FEATURES, RentalNet
from prepare import make_loaders
from train import MAIN_RUN
from train_keras import load_arrays, train_keras_model


def make_inputs(rows=12, seed=0):
    """Random rows with the right number of features."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, N_FEATURES)).astype("float32")


def test_keras_network_returns_one_value_per_row():
    model = build_keras_model(hidden=8)
    predictions = model.predict(make_inputs(rows=7), verbose=0)
    assert predictions.shape == (7, 1)
    assert predictions.flatten().shape == (7,)


@pytest.mark.parametrize("hidden", [1, 4, 16, 64, 512])
def test_parameter_count_matches_pytorch(hidden):
    keras_count = build_keras_model(hidden=hidden).count_params()
    torch_count = sum(p.numel() for p in RentalNet(hidden=hidden).parameters())
    assert keras_count == torch_count


def test_weight_copy_makes_the_networks_predict_the_same():
    torch.manual_seed(1)
    torch_net = RentalNet(hidden=5)
    keras_net = build_keras_model(hidden=5)
    copy_weights_to_keras(torch_net, keras_net)

    x = make_inputs()
    expected = torch_net(torch.tensor(x)).detach().numpy()
    actual = keras_net.predict(x, verbose=0).flatten()
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_keras_and_pytorch_use_the_same_rows():
    loaders = make_loaders()
    arrays = load_arrays()
    for loader, (x, y) in zip(loaders, arrays, strict=True):
        assert torch.equal(torch.from_numpy(x), loader.dataset.features)
        assert torch.equal(torch.from_numpy(y), loader.dataset.targets)


def test_keras_network_can_memorize_one_small_batch():
    keras.utils.set_random_seed(0)
    x = make_inputs(rows=8)
    y = np.random.default_rng(1).normal(scale=5.0, size=8).astype("float32")
    model = build_keras_model(hidden=32)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    before = model.evaluate(x, y, batch_size=8, verbose=0)
    model.fit(x, y, batch_size=8, epochs=500, verbose=0)
    after = model.evaluate(x, y, batch_size=8, verbose=0)
    assert after < 0.05 * before


def test_saved_and_reloaded_model_predicts_the_same(tmp_path):
    model = build_keras_model(hidden=6)
    x = make_inputs()
    path = tmp_path / "model.keras"
    model.save(path)

    reloaded = keras.models.load_model(path)
    np.testing.assert_allclose(
        reloaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6
    )


def test_best_weights_are_restored():
    data = load_arrays()
    x_val, y_val = data[1]
    config = {**MAIN_RUN, "hidden": 8, "lr": 0.1, "epochs": 15}
    model, history, best_epoch = train_keras_model(config, data)

    best_val_loss = history["val_loss"].min()
    assert history.loc[best_epoch - 1, "val_loss"] == best_val_loss
    restored_loss = model.evaluate(x_val, y_val, batch_size=64, verbose=0)[0]
    assert restored_loss == pytest.approx(best_val_loss, rel=1e-3)


def test_same_seed_repeats_a_short_run():
    data = load_arrays()
    config = {**MAIN_RUN, "hidden": 8, "epochs": 3}
    _, first, _ = train_keras_model(config, data, seed=5)
    _, second, _ = train_keras_model(config, data, seed=5)
    np.testing.assert_allclose(first["val_loss"], second["val_loss"], rtol=1e-5)
