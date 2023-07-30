import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


class DataHandler:
    def __init__(self, data_root, batch_size, split_ratio=0.8):
        self.data_root = data_root
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((240, 240)),  # Resize the images to match EfficientNet-B1 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = ImageFolder(root=self.data_root, transform=transform)
        dataset_size = len(dataset)
        split_index = int(self.split_ratio * dataset_size)
        train_dataset, test_dataset = random_split(dataset, [split_index, dataset_size - split_index])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
