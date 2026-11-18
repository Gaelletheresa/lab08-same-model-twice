import pytest
import torch
from torch import nn

from model import N_FEATURES, RentalNet


@pytest.mark.parametrize("hidden", [4, 16, 64, 128])
def test_network_returns_one_value_per_row(hidden):
    model = RentalNet(hidden=hidden)
    batch = torch.randn(10, N_FEATURES)
    assert model(batch).shape == (10,)


def test_one_optimizer_step_changes_every_layer():
    torch.manual_seed(0)
    model = RentalNet(hidden=16)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    x = torch.randn(32, N_FEATURES)
    y = torch.rand(32) * 10

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.MSELoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for name, p in model.named_parameters():
        assert not torch.equal(p, before[name]), name
