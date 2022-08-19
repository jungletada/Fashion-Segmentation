from dataloaders.datasets import SegmentationSet
from torch.utils.data import DataLoader


def make_data_loader(batch_size=16):
    train_set = SegmentationSet.SegmentationDataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader


