
import jsonlines
import pandas as pd

import json
import time
import requests
from requests.exceptions import RequestException
from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
from copy import deepcopy
import random
from loguru import logger
import re
import multiprocessing as mp

def get_tokenize_format(model: str):
    if model == 'qwen':
        origin_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        origin_end = "<|im_end|>\n<|im_start|>assistant\n"
        end_token = '<|endoftext|>'
    elif model == 'llama':
        origin_head = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 01 Dec 2023\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
        origin_end = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        end_token = '<|eot_id|>'
    elif model == 'r1_distill':
        origin_head = '<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{}. '
        origin_end = '<｜Assistant｜><think>\n'
        end_token = '<｜end▁of▁sentence｜>'
    elif model == 'base':
        origin_head = "Question:\n"
        origin_end = "\nAnswer:\nLet's think step by step."
        end_token = None
    else:
        assert 0
    return origin_head, origin_end, end_token

def convert2pq(json_path, parquet_path, n=-1, func=None):
    # Read the JSON file
    df = pd.read_json(json_path, lines=True)
    if n > 0:
        df = df[:n]
    
    if func is not None:
        df = func(df)
    
    # Convert to Parquet format
    df.to_parquet(parquet_path, index=False)

def read_jsonl(file_path: str):
    """读取jsonl文件"""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def keep_last_boxed(text):
    # 找到所有 \boxed{...} 的位置
    pattern = r"\\boxed\{([^{}]*)\}"
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return text  # 如果没有匹配项，直接返回
    
    last_start, last_end = matches[-1].span()  # 最后一个 \boxed{} 的起止位置
    
    # 逐个替换
    result = []
    prev_end = 0
    
    for match in matches:
        start, end = match.span()
        # 添加前面的非匹配部分
        result.append(text[prev_end:start])
        # 如果是最后一个，保留 \boxed{}；否则仅保留内容
        if start == last_start:
            result.append(match.group(0))  # 保留 \boxed{...}
        else:
            result.append(match.group(1))  # 仅保留内容
        prev_end = end
    
    # 添加剩余部分
    result.append(text[prev_end:])
    return ''.join(result)


def save_jsonl(file_path: str, data: list[dict]):
    """保存数据到jsonl文件"""
    with jsonlines.open(file_path, mode='w') as writer:
        for obj in data:
            writer.write(obj)

def process_raw_jsonl(
    raw_dir: str,
    preprocess_dir: str
):
    # 读取raw jsonl, 划分10%作为SFT验证集
    train_data = read_jsonl(f"{raw_dir}/train.jsonl")
    test_data = read_jsonl(f"{raw_dir}/test.jsonl")
    remove_keys = ['unique_id']
    for item in train_data:
        # 删除不需要的key
        for key in remove_keys:
            if key in item:
                del item[key]
    for item in test_data:
        # 删除不需要的key
        for key in remove_keys:
            if key in item:
                del item[key]

    # 随机选择1k例作为SFT train set
    random.seed(1234)
    random.shuffle(train_data)
    sft_train_data = train_data[:900]
    sft_val_data = train_data[900:1000]
    rl_train_data = train_data[1000:]

    # 重新编号
    for i, item in enumerate(test_data):
        item['index'] = i
    
    for i, item in enumerate(sft_train_data):
        item['index'] = i
    
    for i, item in enumerate(sft_val_data):
        item['index'] = i

    for i, item in enumerate(rl_train_data):
        item['index'] = i

    # 保存训练集和验证集
    save_jsonl(f"{preprocess_dir}/prep_sft_train.jsonl", sft_train_data)
    save_jsonl(f"{preprocess_dir}/prep_sft_val.jsonl", sft_val_data)

    save_jsonl(f"{preprocess_dir}/prep_rl_train.jsonl", rl_train_data)
    save_jsonl(f"{preprocess_dir}/prep_rl_test.jsonl", test_data)

        
def generate_batch_inference(
    prep_jsonl_path: str,
    preprocess_dir: str,
    label: str = 'train'
):
    os.makedirs(preprocess_dir, exist_ok=True)
    # 读取raw_jsonl, 生成用于大模型批量推理的jsonl
    data = read_jsonl(prep_jsonl_path)
    result = []
    d_sample = {
        "custom_id": "request-1", 
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-ai/DeepSeek-R1", 
            "messages": [{"role": "user", "content": "How does photosynthesis work?"}], 
            "stream": True, 
            "max_tokens": 4096,
            "temperature": 0.5,
            "top_p": 0.95,
    }}
    gen_start = 'Solve the following math problem in two different approaches. If your answer is not consistent, verify again and choose the answer that is most likely to be correct. Wrap and only wrap your final answer within \\boxed{}. The problem is:\n'
    for b in data:
        prompt = gen_start + b['problem']
        d_sample['body']['messages'][0]['content'] = prompt
        d_sample['custom_id'] =  str(b['index'])
        result.append(deepcopy(d_sample))
    # 生成批量推理的jsonl文件
    save_jsonl(f"{preprocess_dir}/{label}_for_inference.jsonl", result)

def parse_gt_answer(names: list[str], solutions: list[bool]):
    gt_answer = []
    for name, is_knight in zip(names, solutions):
        if bool(is_knight):
            gt_answer.append(f"{name} is a knight")
        else:
            gt_answer.append(f"{name} is a knave")
    gt_answer = ', '.join(gt_answer)
    return gt_answer

def generate_short_cot(
    prep_jsonl_path: str,
    sft_out_dir: str,
    label: str = 'train',
    model: str = 'qwen',
):
    os.makedirs(sft_out_dir, exist_ok=True)
    data = read_jsonl(prep_jsonl_path)
    # 生成SFT数据
    origin_head, origin_end, end_token = get_tokenize_format(model)
    
    
    result = []
    for item in data:
        prompt = origin_head + item['problem'] + origin_end
        response = item['solution'] + end_token
        result.append({
            'id': item['index'],
            'prompt': prompt,
            'response': response,
            'gt_answer': item['answer'],
        })
    
    # 保存数据到jsonl文件
    save_jsonl(f"{sft_out_dir}/{label}.jsonl", result)
    convert2pq(f"{sft_out_dir}/{label}.jsonl", f"{sft_out_dir}/{label}.parquet")


def generate_long_cot_from_batch_inference(
    inference_out_jsonl_path: str,
    prep_jsonl_path: str,
    sft_out_dir: str,
    label: str = 'train',
    check_func=None,
    model:str='qwen'
):
    # 从批量推理结果生成长cot
    os.makedirs(sft_out_dir, exist_ok=True)
    # 配置
    max_tokens = 4096

    inference_data = read_jsonl(inference_out_jsonl_path)

    prep_data = read_jsonl(prep_jsonl_path)
    prep_data = {str(item['index']): item for item in prep_data}

    n_incorrect = 0
    n_toolong = 0
    n_correct = 0

    # 生成SFT数据
    origin_head, origin_end, end_token = get_tokenize_format(model)

    
    result = []
    total_tokens_list = []
    for item in tqdm(inference_data):
        index = item['custom_id']
        total_tokens = item['response']['body']['usage']['total_tokens']
        if total_tokens > max_tokens:
            n_toolong += 1
            continue
        prep_d = prep_data[index]
        gt_answer = prep_d['answer']
        # 解析推理结果
        message = item['response']['body']['choices'][0]['message']
        content = message['content']
        if check_func(content, gt_answer):
            if item['response']['body']['model'] == "deepseek-ai/DeepSeek-R1":
                reasoning_content = message['reasoning_content']
                if not check_func(reasoning_content, gt_answer):
                    reasoning_content += '\nThe final answer is: \\boxed{' + gt_answer + '}.'
                response = reasoning_content + end_token
            else: # V3
                response = content + end_token
            # 去除多余的boxed
            n_correct += 1
            # 生成sft数据
            prompt = origin_head + prep_d['problem'] + origin_end
            
            result.append({
                # 'id': int(item['custom_id']),
                'prompt': prompt,
                'response': response,
                'gt_answer': gt_answer
            })
            total_tokens_list.append(total_tokens)
        else:
            n_incorrect += 1
        
    logger.info(f"label={label}, API调用推理正确: {n_correct}, API调用推理错误: {n_incorrect}, API调用推理超长: {n_toolong}")
    import numpy as np
    # 打乱顺序
    random.seed(1234)
    random.shuffle(result)
    # 重新设置id
    for i, item in enumerate(result):
        item['id'] = i
    
    for percent in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]:
        logger.info(f"label={label}, 总长度{percent}%分位数: {np.percentile(total_tokens_list, percent)}")
    # 保存数据到jsonl文件
    save_jsonl(f"{sft_out_dir}/{label}.jsonl", result)
    # 保存数据为parquet文件
    convert2pq(f"{sft_out_dir}/{label}.jsonl", f"{sft_out_dir}/{label}.parquet")


def generate_data_for_RL(
    raw_jsonl: str,
    rl_output_dir: str,
    label: str,
    model:str
):
    os.makedirs(rl_output_dir, exist_ok=True)
    logger.info(f"label={label}, 生成数据, raw_jsonl={raw_jsonl}")
    origin_head, origin_end, end_token = get_tokenize_format(model)
    
    
    data = read_jsonl(raw_jsonl)
    result = []
    """
    {"answer":"34","gt_answer":"34","target":"34","data_source":"simplelr_qwen","ability":"math","prompt":[{"content":"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.<|im_end|>\n<|im_start|>assistant","role":"user"}],"reward_model":{"ground_truth":"34","style":"rule"},"extra_info":{"answer":"34","index":0,"level":1,"question":"","split":"train"}}
    """
    for d in data:
        d_input = origin_head + d['problem'] + origin_end
        gt_answer = d['answer']
        level = d['level']
        result.append({
            'prompt': [{'content': d_input, 'role': 'user'}],
            'answer': gt_answer,
            'gt_answer': gt_answer,
            'target': gt_answer,
            'data_source': 'math_length',
            'ability': 'math',
            "reward_model":{"ground_truth":gt_answer, "style":"rule"},
            "extra_info":{"answer":gt_answer, "index":None, "level":level, "question":"","split":label},
        })
    
    random.seed(1234)
    random.shuffle(result)
    for i, item in enumerate(result):
        item['extra_info']['index'] = i
    # 保存数据到jsonl文件
    save_jsonl(f"{rl_output_dir}/{label}.jsonl", result)
    # 保存数据为parquet文件
    convert2pq(f"{rl_output_dir}/{label}.jsonl", f"{rl_output_dir}/{label}.parquet")

def generate_data_for_RL_base(
    raw_jsonl: str,
    rl_output_dir: str,
    label: str
):
    os.makedirs(rl_output_dir, exist_ok=True)
    logger.info(f"label={label}, 生成数据, raw_jsonl={raw_jsonl}")
    origin_head, origin_end, eos_token = get_tokenize_format('base')
    data = read_jsonl(raw_jsonl)
    result = []
    for d in data:
        d_input = origin_head + d['problem'] + origin_end
        gt_answer = d['answer']
        level = d['level']
        result.append({
            'prompt': [{'content': d_input, 'role': 'user'}],
            'answer': gt_answer,
            'gt_answer': gt_answer,
            'target': gt_answer,
            'data_source': 'simplelr', # 不是math length, 用于base model
            'ability': 'math',
            "reward_model":{"ground_truth":gt_answer, "style":"rule"},
            "extra_info":{"answer":gt_answer, "index":None, "level":level, "question":"","split":label},
        })
    
    random.seed(1234)
    random.shuffle(result)
    for i, item in enumerate(result):
        item['extra_info']['index'] = i
    # 保存数据到jsonl文件
    save_jsonl(f"{rl_output_dir}/{label}.jsonl", result)
    # 保存数据为parquet文件
    convert2pq(f"{rl_output_dir}/{label}.jsonl", f"{rl_output_dir}/{label}.parquet")

def entry_split_raw():
    os.makedirs('data/Math-Data/preprocessed', exist_ok=True)
    process_raw_jsonl(
        raw_dir='data/Math-Data/raw',
        preprocess_dir='data/Math-Data/preprocessed'
    )

def entry_generate_short_cot(model=['qwen', 'llama']):
    generate_short_cot(
        prep_jsonl_path='data/Math-Data/preprocessed/prep_sft_train.jsonl',
        sft_out_dir=f'data/Math-Data/sft_short_cot_{model}',
        label='train',
        model=model,
    )
    generate_short_cot(
        prep_jsonl_path='data/Math-Data/preprocessed/prep_sft_val.jsonl',
        sft_out_dir=f'data/Math-Data/sft_short_cot_{model}',
        label='val',
        model=model,
    )

def entry_generate_batch_inference():
    generate_batch_inference(
        prep_jsonl_path='data/Math-Data/preprocessed/prep_sft_train.jsonl',
        preprocess_dir='data/Math-Data/preprocessed',
        label='train'
    )
    generate_batch_inference(
        prep_jsonl_path='data/Math-Data/preprocessed/prep_sft_val.jsonl',
        preprocess_dir='data/Math-Data/preprocessed',
        label='val'
    )

def entry_generate_long_cot_from_batch_inference(model=['qwen', 'llama']):
    # NOTE; 这里不使用compute_score_with_length, 因为R1输出不包括special token
    from verl.utils.reward_score.hf_math_verify import compute_score
    def process_check_func(content, gt_answer):
        return compute_score(content, gt_answer)['correctness']

    generate_long_cot_from_batch_inference(
        inference_out_jsonl_path='data/Math-Data/preprocessed/batch_result_val.jsonl',
        prep_jsonl_path='data/Math-Data/preprocessed/prep_sft_val.jsonl',
        sft_out_dir=f'data/Math-Data/sft_long_cot_{model}',
        label='val',
        check_func=process_check_func,
        model=model,
    )
    generate_long_cot_from_batch_inference(
        inference_out_jsonl_path='data/Math-Data/preprocessed/batch_result_train.jsonl',
        prep_jsonl_path='data/Math-Data/preprocessed/prep_sft_train.jsonl',
        sft_out_dir=f'data/Math-Data/sft_long_cot_{model}',
        label='train',
        check_func=process_check_func,
        model=model
    )

def entry_generate_data_for_RL(model=['qwen', 'llama', 'r1_distill']):
    # 为RL生成训练集和测试集
    generate_data_for_RL(
        raw_jsonl='data/Math-Data/preprocessed/prep_rl_test.jsonl',
        rl_output_dir=f'data/Math-Data/RL_data_{model}',
        label='test',
        model=model
    )
    generate_data_for_RL(
        raw_jsonl='data/Math-Data/preprocessed/prep_rl_train.jsonl',
        rl_output_dir=f'data/Math-Data/RL_data_{model}',
        label='train',
        model=model
    )

def entry_generate_RL_for_base():
    # 为base model RL生成训练集和测试集, 不使用特殊token
    generate_data_for_RL_base(
        raw_jsonl='data/Math-Data/preprocessed/prep_rl_test.jsonl',
        rl_output_dir='data/Math-Data/RL_data_base',
        label='test'
    )
    generate_data_for_RL_base(
        raw_jsonl='data/Math-Data/preprocessed/prep_rl_train.jsonl',
        rl_output_dir='data/Math-Data/RL_data_base',
        label='train'
    )

def entry_generate_final_eval_set(model=['qwen', 'llama', 'r1_distill', 'base']):
    assert model in ['qwen', 'r1_distill', 'base']
    def revise_prompt_r1(prompt: str):
        prompt = prompt.removeprefix('<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n')
        prompt = prompt.removesuffix('<|im_end|>\n<|im_start|>assistant\n')
        prompt = '<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{}. ' + prompt + '<｜Assistant｜><think>\n'
        return prompt

    def revise_prompt_base(prompt: str):
        prompt = prompt.removeprefix('<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n')
        prompt = prompt.removesuffix('<|im_end|>\n<|im_start|>assistant\n')
        prompt = 'Question:\n' + prompt + "\nAnswer:\nLet's think step by step."
        return prompt
    
    # 为main_generation生成评估格式数据集
    out_dir = f'data/Math-Data/RL_{model}_eval'
    os.makedirs(out_dir, exist_ok=True)
    pq_path = 'data/Math-Data/RL_data_qwen/test.parquet'
    data = pd.read_parquet(pq_path).to_dict(orient='records')
    result = []
    for item in data:
        prompt = item['prompt'][0]['content']
        gt_answer = item['gt_answer']
        data_source = item['data_source']

        if model == 'r1_distill':
            prompt = revise_prompt_r1(prompt)
        elif model == 'base':
            data_source = 'math_base'
            prompt = revise_prompt_base(prompt)
        elif model == 'qwen':
            prompt = prompt
        else:
            assert 0
        
        
        result.append({
            'prompt': prompt,
            'data_source': data_source,
            'extra_info': {
                'gt_answer': gt_answer,
                'index': item['extra_info']['index'],
                'level': item['extra_info']['level'],
            },
            
        })
    # 保存数据到jsonl文件和pq文件
    save_jsonl(f"{out_dir}/eval.jsonl", result)
    convert2pq(f"{out_dir}/eval.jsonl", f"{out_dir}/eval.parquet")

def entry_generate_cot_from_actor(jsonl_name):
    # 从actor生成结果中取出数据
    
    out_dir = 'data/Math-Data/sft_actor_cot_qwen_1'
    os.makedirs(out_dir, exist_ok=True)
    long_cot_dir = 'data/Math-Data/sft_long_cot_qwen'
    logger.warning("会使用long_cot_dir中的val.jsonl作为验证集, 事先检查格式是否正确")
    from verl.utils.reward_score.hf_math_verify import compute_score_with_length
    # 读取文件
    generated_data = read_jsonl(f'data/Math-Data/actor_generated/{jsonl_name}')
    result = []
    for item in generated_data:
        if compute_score_with_length(item['responses'][0], item['gt_answer'])['correctness']:
            result.append({
                # 'id': int(item['custom_id']),
                'prompt': item['prompt'],
                'response': item['responses'][0],
                'gt_answer': item['gt_answer']
            })
    logger.info(f"Valid samples: {len(result)}/{len(generated_data)}")
    # 保存数据到jsonl文件
    save_jsonl(f"{out_dir}/train.jsonl", result)
    # 保存数据为parquet文件
    convert2pq(f"{out_dir}/train.jsonl", f"{out_dir}/train.parquet")
    # 复制val文件
    import shutil
    shutil.copyfile(f"{long_cot_dir}/val.jsonl", f"{out_dir}/val.jsonl")
    shutil.copyfile(f"{long_cot_dir}/val.parquet", f"{out_dir}/val.parquet")


    

if __name__ == '__main__':
    """
    raw:
    - https://github.com/openai/prm800k/blob/main/prm800k/math_splits/train.jsonl
    - https://github.com/openai/prm800k/blob/main/prm800k/math_splits/test.jsonl
    
    """
    import os

    entry_split_raw()
    entry_generate_short_cot(model='qwen')

    # Generate batch inference data -> remote Deepseek-R1 inference -> collect inference result
    entry_generate_batch_inference()
    # entry_generate_long_cot_from_batch_inference(model='qwen')

    # Generate RL dataset
    entry_generate_data_for_RL(model='qwen')
    entry_generate_RL_for_base()

    # Generate independent eval set
    entry_generate_final_eval_set(model='qwen')
    entry_generate_final_eval_set(model='base')
    # entry_generate_final_eval_set(model='r1_distill')
    
    # Generate Re-distillation dataset
    # entry_generate_cot_from_actor('long-3-new-step-50.jsonl')



