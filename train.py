import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import decode
from utils import compute_cer

from dataset import MarathiDataset
from model import OCRModel
from utils import load_data, create_vocab, SimpleConverter

# ------------------ LOAD DATA ------------------

train_imgs, train_labels = load_data(
    "marathi/file/train_images.txt",
    "marathi/file/train_labels.txt"
)

# Fix paths
BASE_PATH = "marathi/"
train_imgs = [BASE_PATH + p for p in train_imgs]

# Debug check
print(train_imgs[0])
print(os.path.exists(train_imgs[0]))

# ------------------ VOCAB ------------------

vocab = create_vocab(train_labels)
num_classes = len(vocab) + 1  # +1 for blank

converter = SimpleConverter(vocab)

# ------------------ TRANSFORM ------------------       

transform = transforms.Compose([
    transforms.Resize((32, 640)),
    transforms.RandomRotation(3),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------ DATASET ------------------

dataset = MarathiDataset(train_imgs, train_labels, transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ------------------ MODEL ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OCRModel(num_classes).to(device)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)

print("Training started...")
val_imgs, val_labels = next(iter(loader))

# ------------------ TRAIN LOOP ------------------
from tqdm import tqdm

for epoch in range(20):
    loop = tqdm(loader, desc=f"Epoch {epoch}")

    for imgs, labels in loop:

        imgs = imgs.to(device)

        preds = model(imgs)              # (T, B, C)
        preds = preds.log_softmax(2)

        # Encode labels properly
        targets, target_lengths = converter.encode_batch(labels)

        # Move to device
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        input_lengths = torch.full(
            size=(imgs.size(0),),
            fill_value=preds.size(0),
            dtype=torch.long
        ).to(device)

        if (target_lengths == 0).any():
            continue

        if target_lengths.max() > preds.size(0):
            continue

        loss = criterion(
            preds,
            targets,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    model.eval()

    with torch.no_grad():
        sample_imgs = val_imgs.to(device)
        sample_labels = val_labels

        preds = model(sample_imgs)
        preds = preds.log_softmax(2)

        pred_texts = decode(preds, converter)

        print("\n--- Sample Predictions ---")
        n = min(5, len(pred_texts))

        for i in range(n):
            print(f"GT: {sample_labels[i]}")
            print(f"PR: {pred_texts[i]}")
            print("-----")

        cer = compute_cer(pred_texts, sample_labels[:len(pred_texts)])
        print(f"CER: {cer:.4f}")

    model.train()

    epoch_loss = 0

    for imgs, labels in loop:
      ...
      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch} Avg Loss: {avg_loss}")
    print("Max target length:", target_lengths.max().item())
    print("Input length:", preds.size(0))

