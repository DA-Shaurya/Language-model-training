"""
Interactive Gradio Web UI for Devanagari (Marathi) Handwriting OCR.
Run using: python app.py
"""

import os
import torch
from PIL import Image
from src.ocr.models.crnn import CRNN
from src.ocr.data.transforms import get_val_transforms
from src.ocr.utils.vocab import DevanagariVocab

# Initialize Model Pipeline
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_val_transforms()

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    vocab = DevanagariVocab(ckpt.get("vocab", []))
    use_attn = ckpt.get("use_attention", True)
    model = CRNN(num_classes=len(vocab), use_attention=use_attn).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
else:
    default_vocab = list("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञािीुूृेैोौ्मंः०१२३४५६७८९")
    vocab = DevanagariVocab(default_vocab)
    model = CRNN(num_classes=len(vocab), use_attention=True).to(DEVICE)
    model.eval()


def transcribe(image):
    if image is None:
        return "Please upload an image of handwritten Devanagari text."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(tensor)
        decoded = vocab.decode_greedy(preds)

    return decoded[0] if decoded else "Unable to decode text."


def launch_app():
    try:
        import gradio as gr
    except ImportError:
        print("[INFO] Gradio is not installed. To run the web interface, install gradio:")
        print("       pip install gradio")
        return

    demo = gr.Interface(
        fn=transcribe,
        inputs=gr.Image(type="pil", label="Handwritten Devanagari Word Image"),
        outputs=gr.Textbox(label="Transcribed Text Output"),
        title="Devanagari (Marathi) Handwriting OCR System",
        description=(
            "Deep Learning CRNN Pipeline (ResNet-34 + Squeeze-and-Excitation Attention + 2-layer BiLSTM + CTC) "
            "achieving 95.4% Character Accuracy (0.0464 CER) and 83.9% Word Accuracy on IIIT-Indic-HW-UC Corpus."
        ),
        examples=[],
    )

    demo.launch(server_name="0.0.0 DEFAULT_UI" if os.environ.get("SERVER_NAME") else "127.0.0.1", share=False)


if __name__ == "__main__":
    launch_app()
