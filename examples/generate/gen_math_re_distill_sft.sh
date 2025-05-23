#! /bin/bash

USER_ENV=`whoami`
set -x

export HEAD_IP='http://127.0.0.1'
export HEAD_PORT=8265
export N_GPUS_PER_NODE=2

export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1


RUN_NAME="gen_math_sft_from_long"

MODEL_NAME="train_rl_math_long_cot/global_step_50/actor/huggingface"
DATA_PATH="data/Math-Data/sft_long_cot_qwen/train.parquet"
PROMPT_KEY=prompt

BATCH_SIZE=1024

TEMPERATURE=0.0
TOPP=1.0
TOPK=-1
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096
max_num_batched_tokens=16000
max_num_seqs=256

export HDFS_MODEL_PATH='checkpoints'
export HDFS_LOG_PATH='logs'
export RAY_DEDUP_LOGS=0

export HDFS_OUTPUT_PATH='outputs'


export ARNOLD_WORKER_NUM=1
unset http_proxy

export HYDRA_FULL_ERROR=1

ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
  --entrypoint-num-cpus=1 \
  -- python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    data.path=$DATA_PATH \
    data.prompt_key=$PROMPT_KEY \
    data.n_samples=1 \
    data.output_path=$HDFS_OUTPUT_PATH/$RUN_NAME/out.parquet \
    data.batch_size=$BATCH_SIZE \
    data.apply_chat_template=False \
    model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    rollout.temperature=$TEMPERATURE \
    rollout.n=1 \
    rollout.top_p=$TOPP \
    rollout.top_k=$TOPK \
    rollout.prompt_length=$MAX_PROMPT_LENGTH \
    rollout.response_length=$MAX_RESPONSE_LENGTH \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=$max_num_batched_tokens \
    rollout.max_num_seqs=$max_num_seqs