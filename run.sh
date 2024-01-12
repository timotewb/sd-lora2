export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="finetune"
export HUB_MODEL_ID="twb-image-01-512-lora"
export TRAIN_DATA_DIR="data/twb_image_01_512"

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=100 \
  --checkpoints_total_limit=1 \
  --resume_from_checkpoint="latest" \
  --validation_prompt="timwb posing for a photo with a duck" \
  --seed=1337