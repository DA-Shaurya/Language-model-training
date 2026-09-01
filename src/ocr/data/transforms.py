from torchvision import transforms


def get_train_transforms(img_size=(32, 640)):
    """
    Returns targeted data augmentation pipeline for training Devanagari handwriting OCR.
    Includes rotation, color jitter, affine shear, and Gaussian blur to handle handwritten variations.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), shear=2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def get_val_transforms(img_size=(32, 640)):
    """
    Returns clean image transform pipeline for validation and evaluation.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
