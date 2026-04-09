import torch
import Levenshtein

def load_data(img_file, label_file):
    with open(img_file, "r", encoding="utf-8") as f:
        images = f.read().splitlines()

    with open(label_file, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()

    return images, labels


def create_vocab(labels):
    chars = set()
    for word in labels:
        for c in word:
            chars.add(c)
    return sorted(list(chars))


class SimpleConverter:
    def __init__(self, vocab):
        self.char2idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 = blank
        self.idx2char = {i+1: c for i, c in enumerate(vocab)}

    def encode_batch(self, texts):
        targets = []
        lengths = []

        for t in texts:
            encoded = [self.char2idx[c] for c in t if c in self.char2idx]
            targets.extend(encoded)
            lengths.append(len(encoded))

        targets = torch.tensor(targets, dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.long)

        return targets, lengths
    
def decode(preds, converter):
        preds = preds.argmax(2)  # (T, B)
        preds = preds.permute(1, 0)  # (B, T)

        results = []
 
        for seq in preds:
            prev = -1
            text = ""

            for i in seq:
                i = i.item()
                if i != prev and i != 0:
                    text += converter.idx2char.get(i, "")
                prev = i

        results.append(text)

        return results

def compute_cer(preds, gts):
    total_dist = 0
    total_chars = 0

    for p, g in zip(preds, gts):
        total_dist += Levenshtein.distance(p, g)
        total_chars += len(g)

    return total_dist / total_chars if total_chars > 0 else 0