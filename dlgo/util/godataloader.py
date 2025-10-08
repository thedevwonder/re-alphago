import numpy as np
import torch
import torch.utils.data as td

__all__ = [
    'GoDataLoader'
]

class GoDataLoader:
    def __init__(self, feature_path, label_path = None):
        self.feature_path = feature_path
        self.label_path = label_path

    def load_data(self):
        features = np.load(self.feature_path)
        features_tensor = torch.from_numpy(features)

        if self.label_path:
            labels = np.load(self.label_path)
            labels_tensor = torch.from_numpy(labels)
            # this dataset can be directly used with torch's DataLoader
            dataset = td.TensorDataset(features_tensor, labels_tensor)
        else:
            dataset = td.TensorDataset(features_tensor)
        return dataset


