#! /bin/bash

USER_ENV=`whoami`
set -x



export HEAD_IP='http://127.0.0.1'
export HEAD_PORT=8265
export N_GPUS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1


export RUN_NAME='train_sft_math_re_distill_sft'


export VAL_BEFORE_TRAIN='False'
export WANDB_NOTES=""
export REMOVE_PREVIOUS_CKPT=True

export MODEL_NAME="Qwen2.5-1.5B"
export MAX_LENGTH=5000
export TRAIN_BATCH_SIZE=32
export MICRO_TRAIN_BATCH_SIZE=1
export LEARNING_RATE=1e-5
export EPOCHS=2
export SAVE_PER_EPOCH=5
export SCHEDULER="cosine_with_warmup"
export WARMUP_RATIO=0



export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1


export PROJECT_NAME=verl_train
export WANDB_API_KEY='YOUR_API_KEY'
export WANDB_OFFICIAL=1

export HDFS_DATA_PATH="data/Math-Data"
export DATASET_NAME='sft_actor_cot_qwen_1'
export HDFS_MODEL_PATH='pretrain'
export HDFS_CHECKPOINT_PATH='checkpoints'
export HDFS_LOG_PATH='logs'

export LOG_FILE_PATH="$HDFS_LOG_PATH/${RUN_NAME}.log"


export HDFS_OUTPUT_PATH='outputs'


export ARNOLD_WORKER_NUM=1
export WANDB_MODE='offline'
export WANDB_DIR='wandb'
export WANDB_ARTIFACT_DIR='wandb_artifact'


unset http_proxy

echo $WANDB_NOTES


torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS_PER_NODE \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.micro_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/val.parquet \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length=$MAX_LENGTH \
  data.balance_dp_token=False \
  data.apply_chat_template_flag=False \
  data.truncation=error \
  model.partial_pretrain=$HDFS_MODEL_PATH/${MODEL_NAME} \
  model.enable_gradient_checkpointing=True \
  model.use_liger=True \
  optim.lr=$LEARNING_RATE \
  optim.weight_decay=0 \
  optim.betas=[0.9,0.999] \
  optim.scheduler_type=$SCHEDULER \
  optim.warmup_steps_ratio=$WARMUP_RATIO \
  use_remove_padding=False \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/${RUN_NAME} \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.save_model_per_epoch=$SAVE_PER_EPOCH \
  trainer.logger=['console','wandb'] \
  trainer.total_epochs=$EPOCHS 2>&1 | tee $LOG_FILE_PATH



