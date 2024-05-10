import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import DDIMScheduler, ControlNetModel, UNet2DConditionModel
from diffusers.utils import check_min_version
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image

from peft import PeftModel
from control_lora_pipeline import LightControlNetPipeline


base_model = "F:/stable-diffusion-v1-5"

unet = UNet2DConditionModel.from_pretrained(
    base_model, subfolder="unet", torch_dtype=torch.float16
)
controlnet = ControlNetModel.from_unet(unet)

control_lora = PeftModel.from_pretrained(controlnet, "control-lora")
control_lora = control_lora.merge_and_unload()
control_lora.save_pretrained("./save")
