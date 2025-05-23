# Towards Revealing the Effectiveness of Small-Scale Fine-tuning in R1-style Reinforcement Learning

This is the official implementation of paper *Towards Revealing the Effectiveness of Small-Scale Fine-tuning in R1-style Reinforcement Learning*

We provide data and training/evaluation code for reproducing main experiments in paper.

Our code implementation is based on [VeRL](https://github.com/volcengine/verl) and [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason)

## Install Dependencies

**Create Conda Environments**

```bash
conda create -n verl python=3.12.7

conda activate verl
```

```bash
# Install main packages

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

python -m pip install ninja

pip install flash-attn

pip install deepspeed==0.15.2

pip install ray[default]

pip install vllm==0.7.3

pip install tqdm seaborn matplotlib loguru datasets gpustat tensorboard flask latex2sympy2

pip install loralib peft timeout_decorator jsonlines word2number pebble

# Install other packages in requirements.txt
```

## Prepare Dataset

K&K dataset:
- Download `KK-Data`(https://huggingface.co/datasets/Chen-YT/deep-reasoning-kk) and put it under `data/` folder.
- We provide all necessary files but you can also create them from scripts
  - `python data_preprocess/K_and_K_preprocess.py` creates short-CoT/long-CoT/eval dataset
  - `python data_preprocess/KK_re_distillation.py` creates re-distillation dataset
  - Uncomment desired functions in the above scripts.

MATH dataset:
- Download raw train/test file from https://github.com/openai/prm800k/blob/main/prm800k/math_splits
- Put raw files in right place: `data/Math-Data/raw/train.jsonl` and `data/Math-Data/raw/test.jsonl`
- Run `python data_preprocess/Math_preprocess.py ` to create RL/SFT dataset
- Create long CoT dataset:
  - Upload batch inference data in `Math-Data/preprocessed/XX_for_inference.jsonl` to Silicon Flow: https://cloud.siliconflow.cn/batches
  - Put result files in `Math-Data/preprocessed/batch_result_XX.jsonl`. `XX` is `train` or `val`
  - Run `entry_generate_long_cot_from_batch_inference(model='qwen')` in `data_preprocess/Math_preprocess.py`

(Optional) Create Re-distillation dataset:
- Train RL policy and generate responses by `example/generate/gen_math_re_distill_sft`
- Copy jsonl files to `Math-Data/actor_generated/your_jsonl_names.jsonl`
- Run `entry_generate_cot_from_actor('your_jsonl_names.jsonl')` in `data_preprocess/Math_preprocess.py`

## Prepare Model

Download Qwen2.5 models from Hugging Face and put them under `pretrain` folder:
- `pretrain/Qwen2.5-1.5B`
- `pretrain/Qwen2.5-1.5B-instruct`

## Run Scripts

Make sure these folders exist: `outputs`, `ray_temp`, `wandb`, `checkpoints`

For train/eval, you should start ray cluster by `sh start_ray.sh`
- Modify `CUDA_VISIBLE_DEVICES` and `num-gpus` if necessary

Select one script in `example/` and launch scripts by `sh example/XX/YY.sh`. 
- `eval`: evaluate trained models or pretrained models
- `generate`: generate re-distillation dataset from checkpoints
- `train_rl`: RL training
- `train_sft`: SFT training

Modify experiment settings before running script under `examples/` folder:
- Modify `WANDB_API_KEY` in each script.
- Change `N_GPUS_PER_NODE`, `vllm_max_num_seqs`, `ppo_max_token_len_per_gpu`, `log_prob_max_token_len_per_gpu` to fit for your device.
  - Our setting only tested with 2x A800 80G. It needs about 8h for 100 RL steps
  - We use sequence packing, so `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu` will affect performance. But the influence is not significant.
- Remove `WANDB_MODE='offline'` if your cluster can access Internet.
- See replay buffer results in `outputs/RUN_NAME`. See training curves in Wandb.

We use deterministic test script to evaluate performance:
- Change `MODEL_NAME` to evaluate a checkpoint
- See results in `outputs/eval`. It should be deterministic.

## Others

We provide online model outputs in `data/KK-Data/online_generated`

Our modifications on VeRL's core logic are not tested in other experiment settings. This repository should only be used to reproduce results in our paper.

