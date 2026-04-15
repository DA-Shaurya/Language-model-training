# Indic Script OCR (Marathi / Bengali)

A PyTorch-based Deep Learning project for Optical Character Recognition (OCR) of Indic scripts, specifically tailored for Marathi and Bengali word images. This project utilizes a Convolutional Recurrent Neural Network (CRNN) architecture combined with Connectionist Temporal Classification (CTC) loss to transcribe images of text into digital strings.

## Architecture
The model (`model.py`) is built using a standard CRNN approach:
- **CNN Backbone:** Pre-trained `ResNet34` (with the final classification layers removed) for robust feature extraction from input images.
- **RNN Layer:** A 2-layer Bidirectional LSTM (BiLSTM) to capture sequential dependencies in the visual features.
- **Transcription:** A fully connected linear layer followed by `CTCLoss` for alignment-free sequence training.

## Features
- **Custom Dataset Handling:** Dedicated PyTorch `Dataset` class (`MarathiDataset`) for dynamic loading and processing.
- **Data Augmentation:** Implements random rotations and Gaussian blurring to make the model robust against real-world image variations.
- **Evaluation Metrics:** Calculates Character Error Rate (CER) using Levenshtein distance to evaluate model performance.
- **Dynamic Vocabulary:** Automatically generates character mappings based on the provided training labels.

## Project Structure
- `train.py`: Main training loop, data loading, augmentation pipeline, and validation/CER calculation.
- `model.py`: Defines the CRNN (`ResNet34` + `BiLSTM`) neural network architecture.
- `utils.py`: Helper functions for data loading, vocabulary creation, label encoding/decoding, and CER computation.

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- python-Levenshtein
- tqdm

*To install the main dependencies, you can run:*
```bash
pip install torch torchvision python-Levenshtein tqdm
```

## Dataset Setup
The model expects data in a specific directory structure. By default, it looks for a `marathi/` directory (which should be added to `.gitignore` to prevent uploading large files to GitHub):

```text
bengali_ocr/
├── marathi/
│   ├── file/
│   │   ├── train_images.txt    # Text file containing paths to images
│   │   └── train_labels.txt    # Text file containing corresponding ground-truth text
│   └── 1_New_Annoatation_Google/
│       └── 01_marathi_word_images/
│           └── ...             # Actual .jpg image files
├── train.py
├── model.py
└── utils.py
```

## Usage

To train the model, simply run the `train.py` script. Ensure your dataset is properly extracted and placed in the working directory before starting.

```bash
python train.py
```

During training, the script will:
1. Output training loss progress via `tqdm`.
2. Save model checkpoints (`model_epoch_X.pth`) after each epoch.
3. Print sample Ground Truth (GT) vs. Predicted (PR) text for validation.
4. Output the Character Error Rate (CER) to evaluate ongoing performance.

## Acknowledgements
- Built using PyTorch.
- Uses python-Levenshtein for accurate metric evaluation.

## License
This project is licensed under the MIT License.