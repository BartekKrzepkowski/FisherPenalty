from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class TrainerContext:
    epoch_start_at: int = None,
    epoch_end_at: int = None,
    grad_accum_steps: int = None,
    save_multi: int = None,
    log_multi: int = None,
    clip_value: float = None,
    base_path: str = None,
    exp_name: str = None,
    logger_config: Dict = None,
    whether_disable_tqdm: bool = None,
    random_seed: int = None,
    device: torch.device = None
