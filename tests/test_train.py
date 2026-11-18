import copy

import torch
from torch import nn

from model import RentalNet
from prepare import make_loaders
from train import run_epoch, train_model


def first_batch(rows=32):
    train_loader, _, _ = make_loaders()
    features, targets = next(iter(train_loader))
    return features[:rows], targets[:rows]


def test_network_can_memorize_one_small_batch():
    torch.manual_seed(0)
    batch = [first_batch()]
    model = RentalNet(hidden=64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start, _ = run_epoch(model, batch, loss_fn, optimizer)
    for _ in range(300):
        end, _ = run_epoch(model, batch, loss_fn, optimizer)

    assert end < 0.05 * start


def test_gradients_are_cleared_between_steps():
    torch.manual_seed(0)
    batch = first_batch()
    model = RentalNet(hidden=16)
    loss_fn = nn.MSELoss()

    model.zero_grad()
    loss_fn(model(batch[0]), batch[1]).backward()
    expected = [p.grad.clone() for p in model.parameters()]

    # With lr=0 the weights never move, so every step sees the same gradient.
    # If zero_grad() is missing, the gradients add up and become 2x as large.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    run_epoch(model, [batch, batch], loss_fn, optimizer)

    for p, grad in zip(model.parameters(), expected, strict=True):
        assert torch.allclose(p.grad, grad)


def test_validation_epoch_leaves_weights_unchanged():
    _, val_loader, _ = make_loaders()
    model = RentalNet(hidden=16)
    before = copy.deepcopy(model.state_dict())

    run_epoch(model, val_loader, nn.MSELoss())

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_train_model_logs_every_epoch_and_restores_best_weights():
    torch.manual_seed(0)
    train_loader, val_loader, _ = make_loaders()
    model = RentalNet(hidden=16)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, epochs=5
    )

    assert len(history) == 5
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "val_mae"}
    best_row = min(history, key=lambda row: row["val_loss"])
    assert best_epoch == best_row["epoch"]

    val_loss, _ = run_epoch(model, val_loader, loss_fn)
    assert abs(val_loss - best_row["val_loss"]) < 1e-4
