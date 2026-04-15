import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MarathiDataset
from model import OCRModel
from utils import load_data, create_vocab, SimpleConverter, decode, compute_cer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_PATH   = "marathi/"
BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 3e-4
GRAD_CLIP   = 5.0
VAL_SPLIT   = 0.1
NUM_WORKERS = 0        # Must stay 0 on Windows — multiprocessing requires __main__ guard
SAVE_DIR    = "checkpoints"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────
    train_imgs, train_labels = load_data(
        "marathi/file/train_images.txt",
        "marathi/file/train_labels.txt",
    )
    train_imgs = [BASE_PATH + p for p in train_imgs]
    print(train_imgs[0], "→ exists:", os.path.exists(train_imgs[0]))

    # ─────────────────────────────────────────
    # VOCAB
    # ─────────────────────────────────────────
    vocab       = create_vocab(train_labels)
    num_classes = len(vocab) + 1   # +1 for CTC blank (index 0)
    converter   = SimpleConverter(vocab)
    print(f"Vocab size: {len(vocab)}  |  num_classes: {num_classes}")

    # ─────────────────────────────────────────
    # TRANSFORMS  (augmented for train, clean for val)
    # ─────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((32, 640)),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), shear=2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ─────────────────────────────────────────
    # TRAIN / VAL SPLIT
    # ─────────────────────────────────────────
    n_total   = len(train_imgs)
    n_val     = max(1, int(n_total * VAL_SPLIT))
    train_idx = list(range(n_total - n_val))
    val_idx   = list(range(n_total - n_val, n_total))

    train_dataset = MarathiDataset(
        [train_imgs[i] for i in train_idx],
        [train_labels[i] for i in train_idx],
        transform=train_transform,
    )
    val_dataset = MarathiDataset(
        [train_imgs[i] for i in val_idx],
        [train_labels[i] for i in val_idx],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    # ─────────────────────────────────────────
    # MODEL / LOSS / OPTIMISER / SCHEDULER
    # ─────────────────────────────────────────
    model     = OCRModel(num_classes, dropout=0.3).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_cer   = float("inf")
    best_epoch = -1

    # ─────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        skipped    = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for imgs, labels in loop:
            imgs      = imgs.to(device)
            preds     = model(imgs)           # (T, B, C)
            preds_log = preds.log_softmax(2)

            targets, target_lengths = converter.encode_batch(labels)
            targets        = targets.to(device)
            target_lengths = target_lengths.to(device)

            input_lengths = torch.full(
                (imgs.size(0),), preds_log.size(0), dtype=torch.long, device=device
            )

            if (target_lengths == 0).any() or target_lengths.max() > preds_log.size(0):
                skipped += 1
                continue

            loss = criterion(preds_log, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(train_loader) - skipped)
        print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Skipped: {skipped}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth"))

        # ── VALIDATION ────────────────────────
        model.eval()
        all_preds, all_gts = [], []

        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs   = val_imgs.to(device)
                preds      = model(val_imgs).log_softmax(2)
                pred_texts = decode(preds, converter)
                all_preds.extend(pred_texts)
                all_gts.extend(val_labels)

        cer = compute_cer(all_preds, all_gts)
        print(f"Epoch {epoch+1} | Val CER: {cer:.4f}")

        print("\n--- Sample Predictions ---")
        for i in range(min(5, len(all_preds))):
            print(f"  GT: {all_gts[i]}")
            print(f"  PR: {all_preds[i]}")
            print("  -----")

        if cer < best_cer:
            best_cer, best_epoch = cer, epoch + 1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ✓ New best model saved (CER={best_cer:.4f})")

        scheduler.step(cer)
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}\n")

    print(f"\nTraining complete. Best CER: {best_cer:.4f} at epoch {best_epoch}.")
    print(f"Best model → {os.path.join(SAVE_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()