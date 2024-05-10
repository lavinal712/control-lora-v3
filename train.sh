PEFT_TYPE="lora"
LORA_RANK=32
DATASET_NAME="/mnt/hongyuqian/coco-stuff-captioned"
PROJECT_NAME="control_${PEFT_TYPE}"
RUN_NAME="${PEFT_TYPE}_${LORA_RANK}"
MODEL_NAME="/mnt/hongyuqian/stable-diffusion-v1-5"
OUTPUT_DIR="/mnt/hongyuqian/output/${DATASET_NAME}/${RUN_NAME}"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
CUDA_VISIBLE_DEVICES=3 accelerate launch train_control_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" \
  --dataset_name=$DATASET_NAME \
  --conditioning_image_column="guide" \
  --image_column="image" \
  --caption_column="caption" \
  --resolution=512 \
  --gradient_accumulation_steps=3 \
  --gradient_checkpointing \
  --learning_rate=1e-5 \
  --checkpointing_steps=500 \
  --max_train_steps=75000 \
  --validation_steps=5000 \
  --num_validation_samples=3 \
  --num_validation_images=12 \
  --train_batch_size=4 \
  --dataloader_num_workers=2 \
  --seed=0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --wandb_project_name=$PROJECT_NAME \
  --wandb_run_name=$RUN_NAME \
  --enable_xformers_memory_efficient_attention \
  --use_lora \
  --lora_r=$LORA_RANK \
  --lora_bias="lora_only" \
