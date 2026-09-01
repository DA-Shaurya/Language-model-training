from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    num_classes: int = 118  # 117 character vocabulary + 1 CTC blank token (index 0)
    img_height: int = 32
    img_width: int = 640
    hidden_size: int = 256
    num_rnn_layers: int = 2
    dropout: float = 0.3
    use_attention: bool = True  # Toggle for ablation studies (SE Attention module)


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    val_split: float = 0.1
    num_workers: int = 0  # Safe default for Windows multiprocessing
    save_dir: str = "checkpoints"
    use_amp: bool = True
    seed: int = 42
