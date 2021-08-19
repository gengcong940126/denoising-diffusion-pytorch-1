from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from tensorfn.data import LMDBReader
from torchvision.datasets import CIFAR10, STL10, CelebA
from torchvision.datasets import ImageFolder

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform,train, dataset, resolution=256):
        if dataset == 'cifar10':
            self.reader = CIFAR10(root=path,
                                train=train,
                                download=True)
        else:
            self.reader = LMDBReader(path, reader="raw")

        self.resolution = resolution
        self.transform = transform
        self.dataset = dataset
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        if self.dataset in ['cifar10','animeface']:
            img, label = self.reader[index]
            img, label = self.transform(img), int(label)
        else:
            img_bytes = self.reader.get(
                f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            )

            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
            img = self.transform(img)

        return img
