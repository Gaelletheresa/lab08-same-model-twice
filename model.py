from torch import nn

N_FEATURES = 8


class RentalNet(nn.Module):
    """A small network: two hidden layers with ReLU, one output, no activation."""

    def __init__(self, n_features=N_FEATURES, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


if __name__ == "__main__":
    from prepare import make_loaders

    train_loader, _, _ = make_loaders()
    features, targets = next(iter(train_loader))
    model = RentalNet()
    predictions = model(features)
    print("features:   ", features.shape)
    print("predictions:", predictions.shape)
    print("targets:    ", targets.shape)
