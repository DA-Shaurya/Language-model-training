from PIL import Image
from torch.utils.data import Dataset

class MarathiDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Skip empty labels
        if label.strip() == "":
            return self.__getitem__((idx + 1) % len(self))

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Skip broken images
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label