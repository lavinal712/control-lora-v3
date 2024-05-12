import torch

import diffusers
from diffusers import UNet2DConditionModel, ControlNetModel
from peft import PeftModel


base_model = "F:/stable-diffusion-v1-5"
controlnet_save_path = "./save"

unet = UNet2DConditionModel.from_pretrained(
    base_model, subfolder="unet", torch_dtype=torch.float16
)

controlnet = ControlNetModel.from_unet(unet)
controlnet = PeftModel.from_pretrained(controlnet, "controlnet")
controlnet = controlnet.merge_and_unload()
controlnet.save_pretrained(controlnet_save_path)
