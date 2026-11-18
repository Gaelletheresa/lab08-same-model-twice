import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import STATION_TYPES, TARGET, make_rentals

SPLIT_SEED = 2026
NUMERIC_COLUMNS = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORY_COLUMN = "station_type"


def split_frame(frame, seed=SPLIT_SEED):
    """Split the table into train (60%), validation (20%) and test (20%) rows."""
    train, rest = train_test_split(frame, test_size=0.4, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def prepare_features(train, val, test):
    """Fit the scaler and encoder on the training rows only, then apply them."""
    scaler = StandardScaler().fit(train[NUMERIC_COLUMNS])
    encoder = OneHotEncoder(categories=[STATION_TYPES], sparse_output=False)
    encoder.fit(train[[CATEGORY_COLUMN]])

    def transform(frame):
        numbers = scaler.transform(frame[NUMERIC_COLUMNS])
        categories = encoder.transform(frame[[CATEGORY_COLUMN]])
        features = np.hstack([numbers, categories]).astype(np.float32)
        targets = frame[TARGET].to_numpy().astype(np.float32)
        return features, targets

    x_train, y_train = transform(train)
    x_val, y_val = transform(val)
    x_test, y_test = transform(test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class BikeDataset(Dataset):
    """Returns one row at a time: (features, target) as float32 tensors."""

    def __init__(self, features, targets):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def make_loaders(batch_size=64, seed=SPLIT_SEED):
    """Build the train, validation and test loaders. Only train shuffles."""
    frame = make_rentals()
    train, val, test = split_frame(frame, seed)
    train_data, val_data, test_data = prepare_features(train, val, test)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BikeDataset(*train_data),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(BikeDataset(*val_data), batch_size=batch_size)
    test_loader = DataLoader(BikeDataset(*test_data), batch_size=batch_size)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_loaders()
    features, targets = next(iter(train_loader))
    print("features:", features.shape, features.dtype)
    print("targets: ", targets.shape, targets.dtype)
    print("batches:", len(train_loader), len(val_loader), len(test_loader))
