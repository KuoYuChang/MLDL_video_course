from torch.utils.data import Dataset


class playDataset(Dataset):
    def __init__(self, labels, data):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_ith = self.data[idx]
        label = self.labels[idx]

        return data_ith, label