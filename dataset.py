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
        # Use a visited set to avoid infinite recursion on a mostly-broken dataset
        for attempt in range(len(self)):
            i = (idx + attempt) % len(self)
            img_path = self.image_paths[i]
            label = self.labels[i]

            if label.strip() == "":
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            if self.transform:
                image = self.transform(image)

            return image, label

        raise RuntimeError("No valid samples found in the entire dataset.")