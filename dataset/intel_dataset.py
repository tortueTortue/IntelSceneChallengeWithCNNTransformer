# https://www.kaggle.com/puneet6060/intel-image-classification

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


TRAIN_FOLDER_PATH = "E:/Image Datasets/Intel Scenes/archive/seg_train/seg_train"
TEST_FOLDER_PATH = "E:/Image Datasets/Intel Scenes/archive/seg_test/seg_test"

def load_intel_scene_data(batch=10, root_path="E:/Image Datasets/Intel Scenes/"):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ImageFolder(root=TRAIN_FOLDER_PATH, transform=transform)
    
    torch.manual_seed(43)
    train_dataset, val_dataset = random_split(dataset, [11928, len(dataset) - 11928])

    train_data_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,  num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,  num_workers=4)

    test_data = ImageFolder(root=TEST_FOLDER_PATH, transform=transform)
    test_data_loader  = DataLoader(test_data, batch_size=batch, shuffle=True, num_workers=4)

    return train_data_loader, val_data_loader, test_data_loader


class IntelSceneDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_folder, root_dir, transform=None):
        """
        Args:
            image_folder (string): Path to the folder with annotations.

        """
        pass

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        return sample

if __name__ == "__main__":
    tr, va, te = load_intel_scene_data()

    images, labels = iter(tr).next()

    plt.imshow(images[0][0].numpy())
    plt.show()