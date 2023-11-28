"""Find the best checkpoint if training script died before saving the model."""
import os
from transformers.trainer_callback import TrainerState

save_dir = "../data/vit-24-10"
ckpt_dirs = os.listdir(save_dir)
ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
last_ckpt = ckpt_dirs[-1]

state = TrainerState.load_from_json(f"{save_dir}/{last_ckpt}/trainer_state.json")
