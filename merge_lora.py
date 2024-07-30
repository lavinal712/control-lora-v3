import argparse

from diffusers import UNet2DConditionModel, ControlNetModel
from peft import PeftModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--control_lora",
        type=str,
        default=None,
        required=True,
        help="Path to control-lora model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to save ControlNet model.",
    )

    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    controlnet = PeftModel.from_pretrained(controlnet, args.control_lora)
    controlnet = controlnet.merge_and_unload()
    controlnet.save_pretrained(args.output_dir)
