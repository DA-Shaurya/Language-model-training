import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.ocr.models.crnn import CRNN
from src.ocr.data.dataset import IIITIndicHWDataset
from src.ocr.data.transforms import get_train_transforms, get_val_transforms
from src.ocr.utils.vocab import DevanagariVocab, create_vocab
from src.ocr.metrics.evaluator import evaluate_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Devanagari Handwriting OCR Model (ResNet34 + SE-Attention + BiLSTM + CTC)"
    )
    parser.add_argument("--data-dir", type=str, default="marathi", help="Path to dataset directory")
    parser.add_argument("--train-images", type=str, default="marathi/file/train_images.txt")
    parser.add_argument("--train-labels", type=str, default="marathi/file/train_labels.txt")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping norm")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set split fraction")
    parser.add_argument("--no-attention", action="store_true", help="Disable SE-Attention (ablation baseline)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Run a fast 1-batch dry run for testing")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()


def load_file_list(file_path: str):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_synthetic_demo_data(data_dir: str):
    os.makedirs(os.path.join(data_dir, "file"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dummy_images"), exist_ok=True)

    dummy_images = []
    dummy_labels = ["मराठी", "देवनागरी", "भारत", "गणेश", "महाराष्ट्र"] * 10

    for i, label in enumerate(dummy_labels):
        rel_path = f"dummy_images/sample_{i}.jpg"
        full_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(full_path):
            img = Image.new("RGB", (640, 32), color=(245, 245, 245))
            img.save(full_path)
        dummy_images.append(rel_path)

    return dummy_images, dummy_labels


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | SE-Attention: {not args.no_attention}")

    # Load dataset index files
    raw_images = load_file_list(args.train_images)
    raw_labels = load_file_list(args.train_labels)

    if not raw_images or not raw_labels:
        print(f"[WARNING] Dataset text files not found at {args.train_images}. Generating synthetic demo dataset.")
        raw_images, raw_labels = generate_synthetic_demo_data(args.data_dir)

    # Ensure paths prefix
    image_paths = [img if img.startswith(args.data_dir) else os.path.join(args.data_dir, img) for img in raw_images]

    # Create Vocabulary
    vocab_list = create_vocab(raw_labels)
    vocab = DevanagariVocab(vocab_list)
    num_classes = len(vocab)
    print(f"[INFO] Vocabulary Size: {len(vocab_list)} characters | Total Classes (with CTC blank): {num_classes}")

    # Transforms & Datasets
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    n_total = len(image_paths)
    n_val = max(1, int(n_total * args.val_split))
    train_idx = list(range(n_total - n_val))
    val_idx = list(range(n_total - n_val, n_total))

    train_dataset = IIITIndicHWDataset(
        [image_paths[i] for i in train_idx],
        [raw_labels[i] for i in train_idx],
        transform=train_tf,
    )
    val_dataset = IIITIndicHWDataset(
        [image_paths[i] for i in val_idx],
        [raw_labels[i] for i in val_idx],
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # Model, Loss, Optimizer
    model = CRNN(num_classes=num_classes, use_attention=not args.no_attention).to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    use_amp = device.type in ["cuda", "cpu"]
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_cer = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        skipped = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                preds = model(imgs)  # (T, B, num_classes)
                preds_log = preds.log_softmax(2)

                targets, target_lengths = vocab.encode_batch(labels)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = torch.full((imgs.size(0),), preds_log.size(0), dtype=torch.long, device=device)

                if (target_lengths == 0).any() or target_lengths.max() > preds_log.size(0):
                    skipped += 1
                    continue

                loss = criterion(preds_log, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if args.dry_run:
                print("\n[DRY RUN] Passed 1 batch dry-run successfully.")
                return

        avg_loss = epoch_loss / max(1, len(train_loader) - skipped)

        # Validation Phase
        model.eval()
        all_preds, all_gts = [], []

        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs = val_imgs.to(device)
                preds = model(val_imgs)
                pred_texts = vocab.decode_greedy(preds)
                all_preds.extend(pred_texts)
                all_gts.extend(val_labels)

        metrics = evaluate_metrics(all_preds, all_gts)
        cer = metrics["cer"]
        char_acc = metrics["char_accuracy"] * 100
        word_acc = metrics["word_accuracy"] * 100

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} | Avg Loss: {avg_loss:.4f} | "
            f"Val CER: {cer:.4f} | Char Acc: {char_acc:.2f}% | Word Acc: {word_acc:.2f}%"
        )

        if cer < best_cer:
            best_cer = cer
            best_epoch = epoch + 1
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "vocab": vocab.vocab,
                    "cer": cer,
                    "word_acc": word_acc,
                    "use_attention": not args.no_attention,
                },
                best_model_path,
            )
            print(f"  [SAVED] New best model checkpoint saved to {best_model_path}")

        scheduler.step(cer)

    print(f"\n[DONE] Training complete. Best CER: {best_cer:.4f} (Epoch {best_epoch}).")


if __name__ == "__main__":
    main()