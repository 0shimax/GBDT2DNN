from pathlib import Path
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, features, labels, gbdt_model, transform=None, is_train=False):
        super().__init__()
        self.transform = transform
        self.is_train = is_train
        self.features = features
        self.labels = labels
        self.gbdt_model = gbdt_model

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        feature_for_dnn = self.gbdt_model.apply(feature.reshape(1, -1))
        return torch.LongTensor([feature_for_dnn]), torch.LongTensor([label])


def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader
