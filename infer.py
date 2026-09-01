import argparse
import os
from PIL import Image
import torch

from src.ocr.models.crnn import CRNN
from src.ocr.data.transforms import get_val_transforms
from src.ocr.utils.vocab import DevanagariVocab


def parse_args():
    parser = argparse.ArgumentParser(description="Devanagari Handwritten Word OCR Inference CLI")
    parser.add_argument("--image", type=str, help="Path to single input handwritten word image")
    parser.add_argument("--dir", type=str, help="Path to directory containing input images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    return parser.parse_args()


class DevanagariOCRInference:
    """
    Inference Pipeline for Devanagari (Marathi) Word OCR.
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_val_transforms()

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            vocab_list = checkpoint.get("vocab", [])
            self.vocab = DevanagariVocab(vocab_list)
            use_attention = checkpoint.get("use_attention", True)

            self.model = CRNN(num_classes=len(self.vocab), use_attention=use_attention).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            print(f"[INFO] Successfully loaded model from '{checkpoint_path}'. Vocab size: {len(vocab_list)}")
        else:
            print(f"[WARNING] Checkpoint '{checkpoint_path}' not found. Initializing unweighted model for demonstration.")
            # Default Devanagari vocabulary sample
            default_vocab = list("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञािीुूृेैोौ्मंः०१२३४५६७८९")
            self.vocab = DevanagariVocab(default_vocab)
            self.model = CRNN(num_classes=len(self.vocab), use_attention=True).to(self.device)
            self.model.eval()

    def predict_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return f"[ERROR] File not found: {image_path}"

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)
            decoded = self.vocab.decode_greedy(preds)

        return decoded[0] if decoded else ""


def main():
    args = parse_args()

    ocr = DevanagariOCRInference(args.checkpoint)

    if args.image:
        result = ocr.predict_image(args.image)
        print(f"\nImage: {args.image}\nRecognized Text: {result}")
    elif args.dir and os.path.exists(args.dir):
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"\nProcessing {len(files)} images in '{args.dir}':")
        for f in files[:20]:
            pred = ocr.predict_image(f)
            print(f"  [{os.path.basename(f)}]: {pred}")
    else:
        print("Please specify --image <path> or --dir <folder_path> for inference.")


if __name__ == "__main__":
    main()
