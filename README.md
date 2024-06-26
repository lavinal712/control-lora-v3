# ControlLoRA Version 3: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 3

ControlLoRA Version 3 is a neural network structure extended from [ControlNet](https://github.com/lllyasviel/ControlNet) to control diffusion models by adding extra conditions.

Inspired by [ControlLoRA](https://github.com/HighCWu/ControlLoRA), [control-lora-v2](https://github.com/HighCWu/control-lora-v2) and script [train_controlnet.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) from [diffusers](https://github.com/huggingface/diffusers), [control-lora-v3](https://github.com/lavinal712/control-lora-v3) does not add new features, but provides a [PEFT](https://github.com/huggingface/peft) implement of ControlLoRA.

## News

- [x] Jun. 08, 2024. Norm layer is trainable. 
- [x] May. 19, 2024. Add [DoRA](https://arxiv.org/abs/2402.09353).

## Data

To train ControlLoRA, you should have image-conditioning_image-text datasets. Of course you can hardly train on [LAION-5B](https://openxlab.org.cn/datasets/OpenDataLab/LAION-5B) dataset in direct like Stable Diffusion. Here are some:

- [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k). I do not suggest you to train ControlLoRA seriously as it is simple and lack of diversity.
- [HighCWu/diffusiondb_2m_first_5k_canny](https://huggingface.co/datasets/HighCWu/diffusiondb_2m_first_5k_canny). A small canny dataset. Here is [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) dataset. For canny condition, you can easily generate your own dataset.
- [Nahrawy/VIDIT-Depth-ControlNet](https://huggingface.co/datasets/Nahrawy/VIDIT-Depth-ControlNet). Depth map? Heat map? But it is good!
- [SaffalPoosh/scribble_controlnet_dataset](https://huggingface.co/datasets/SaffalPoosh/scribble_controlnet_dataset). Many duplicate images. I suggest you synthesize your dataset.

## Model

[Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) is the base model.

[Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [Stable Diffusion v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) need to be vertified.

[Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) needs to be vertified, but probably does not work.

## Train

You can train either ControlNet or ControlLoRA using script [train_control_lora.py](https://github.com/lavinal712/control-lora-v3/blob/main/train_control_lora.py).

### Train ControlNet

By observation, training 50000 steps with batch size of 4 is the balance between image quality, control ability and time.

```bash
accelerate launch train_control_lora.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="controlnet-model" \
 --dataset_name="fusing/fill50k" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --max_train_steps=100000 \
 --tracker_project_name="controlnet" \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --report_to wandb
```

### Train ControlLoRA

To train ControlLoRA, add `--use_lora` in start command to activate it.

```bash
accelerate launch train_control_lora.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="control_lora-model" \
 --dataset_name="fusing/fill50k" \
 --resolution=512 \
 --learning_rate=1e-4 \
 --train_batch_size=4 \
 --max_train_steps=100000 \
 --tracker_project_name="control_lora" \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --report_to wandb \
 --use_lora \
 --lora_r=32
```

You can also train ControlLoRA / ControlNet with your own dataset.

```bash
accelerate launch train_control_lora.py \
 --pretrained_model_name_or_path="stable-diffusion-v1-5" \
 --output_dir="control_lora-model" \
 --conditioning_image_column="hint" \
 --image_column="jpg" \
 --caption_column="txt" \
 --resolution=512 \
 --learning_rate=1e-4 \
 --train_batch_size=4 \
 --num_train_epochs=3 \
 --max_train_steps=100000 \
 --tracker_project_name="control_lora" \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --report_to wandb \
 --use_lora \
 --lora_r=32 \
 --custom_dataset="fill50k"
```

## Merge

If you want to train ControlNet, you have already got it. If you got a lora, merge it!

```bash
python merge_lora.py
```

## Test

Original image:

![house](assets/house.png)

Output:

![house_grid](assets/house_grid.png)

## Citation

    @software{lavinal7122024controllorav3,
        author = {lavinal712},
        month = {5},
        title = {{ControlLoRA Version 3: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 3}},
        url = {https://github.com/lavinal712/control-lora-v3},
        version = {1.0.0},
        year = {2024}
    }
