
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
    raw_jsonl_paths: list[str],
    preprocess_dir: str
):
    # 读取raw jsonl, 划分10%作为SFT验证集
   
    data = []
    for raw_jsonl_path in raw_jsonl_paths:
        data += read_jsonl(raw_jsonl_path)

    # 随机选择10%作为验证集
    random.seed(1234)
    random.shuffle(data)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    valid_data = data[int(n * 0.9):]
    # 重新编号
    for i, item in enumerate(train_data):
        item['index'] = i
    for i, item in enumerate(valid_data):
        item['index'] = i
    # 保存训练集和验证集
    save_jsonl(f"{preprocess_dir}/prep_train.jsonl", train_data)
    save_jsonl(f"{preprocess_dir}/prep_valid.jsonl", valid_data)
        
def generate_batch_inference(
    prep_jsonl_path: str,
    preprocess_dir: str,
    label: str = 'train'
):
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
            "temperature": 0.0,
            # "top_p": 0.95,
    }}
    gen_end = ' Your final answer should be in this format: \\boxed{XX is a knave, YY is a knight, ...}'
    for b in data:
        prompt = b['quiz'] + gen_end
        d_sample['body']['messages'][0]['content'] = prompt
        d_sample['custom_id'] =  str(b['index'])
        result.append(deepcopy(d_sample))
    # 生成批量推理的jsonl文件
    save_jsonl(f"{preprocess_dir}/{label}_for_inference.jsonl", result)

def generate_batch_inference_from_raw(
    raw_jsonl_path: str,
    out_dir: str,
    label: str = 'test',
    model_name: str = "deepseek-ai/DeepSeek-R1"
):
    if model_name == "deepseek-ai/DeepSeek-R1":
        max_tokens = 16384
    elif model_name == 'Qwen/QwQ-32B':
        max_tokens = 16384 # 不确定

    # 读取raw_jsonl, 生成用于大模型批量推理的jsonl
    data = read_jsonl(raw_jsonl_path)
    os.makedirs(out_dir, exist_ok=True)
    result = []
    d_sample = {
        "custom_id": "request-1", 
        "method": "POST", 
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name, 
            "messages": [{"role": "user", "content": "How does photosynthesis work?"}], 
            "stream": True, 
            "max_tokens": max_tokens,
            "temperature": 0.0,
            # "top_p": 0.95,
    }}
    gen_end = ' Your final answer should be in this format: \\boxed{XX is a knave, YY is a knight, ...}'
    for b in data:
        prompt = b['quiz'] + gen_end
        d_sample['body']['messages'][0]['content'] = prompt
        d_sample['custom_id'] =  str(b['index'])
        result.append(deepcopy(d_sample))
    # 生成批量推理的jsonl文件
    save_jsonl(f"{out_dir}/{label}_for_inference.jsonl", result)

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
    label: str = 'train'
):
    data = read_jsonl(prep_jsonl_path)
    # 生成SFT数据
    origin_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
    gen_end = ' Your final answer should be in this format: \\boxed{XX is a knave, YY is a knight, ...}'
    origin_end = "<|im_end|>\n<|im_start|>assistant"
    result = []
    for item in data:
        cot_answer = item['cot_head'] + ' ' +  ' '.join(item['cot_repeat_steps']) + ' ' + item['cot_foot']
        gt_answer = parse_gt_answer(item['names'], item['solution'])
        
        cot_result = 'The final answer is: \\boxed{' + gt_answer + '}.<|endoftext|>'
        prompt = origin_head + item['quiz'] + gen_end + origin_end
        response = '\n' + cot_answer + '\n' + cot_result
        result.append({
            'id': item['index'],
            'prompt': prompt,
            'response': response,
            'gt_answer': gt_answer
        })
    # 保存数据到jsonl文件
    save_jsonl(f"{sft_out_dir}/{label}.jsonl", result)
    # 保存数据为parquet文件
    convert2pq(f"{sft_out_dir}/{label}.jsonl", f"{sft_out_dir}/{label}.parquet")


def generate_long_cot_from_batch_inference(
    inference_out_jsonl_path: str,
    prep_jsonl_path: str,
    sft_out_dir: str,
    label: str = 'train',
    check_func=None
):
    # 从批量推理结果生成长cot

    # 配置
    max_tokens = 4000

    inference_data = read_jsonl(inference_out_jsonl_path)

    prep_data = read_jsonl(prep_jsonl_path)
    prep_data = {str(item['index']): item for item in prep_data}

    n_incorrect = 0
    n_toolong = 0
    n_correct = 0
    # 生成SFT数据
    origin_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
    gen_end = ' Your final answer should be in this format: \\boxed{XX is a knave, YY is a knight, ...}'
    origin_end = "<|im_end|>\n<|im_start|>assistant"
    result = []
    total_tokens_list = []
    for item in inference_data:
        index = item['custom_id']
        total_tokens = item['response']['body']['usage']['total_tokens']
        if total_tokens > max_tokens:
            n_toolong += 1
            continue
        prep_d = prep_data[index]
        gt_answer = parse_gt_answer(prep_d['names'], prep_d['solution'])
        # 解析推理结果
        message = item['response']['body']['choices'][0]['message']
        content, reasoning_content = message['content'], message['reasoning_content']
        if check_func(content, gt_answer) == 1:
            if check_func(reasoning_content, gt_answer) == 0: # reason部分没有输出boxed
                reasoning_content += '\nFinally, write the answer in the boxed format: \\boxed{' + gt_answer + '}.'
            # 去掉除了最后一个以外的boxed
            # import re
            # boxed_pattern = re.compile(r"\\boxed\{([^{}]*)\}")
            # boxed_match = boxed_pattern.findall(reasoning_content)
            # if len(boxed_match) > 1:
            #     logger.debug(f"label={label}, index={index}, boxed_match={boxed_match}")
            # reasoning_content = keep_last_boxed(reasoning_content)
            # 去除多余的boxed
            n_correct += 1
            # 生成sft数据
            prompt = origin_head + prep_d['quiz'] + gen_end + origin_end
            response =  '\n' + reasoning_content + '<|endoftext|>'
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


def eval_baseline(
    jsonl_paths: dict[int, str],
    raw_paths: dict[int, str],
    out_dir: str,
    max_token_detection: int = 16384,
    label: str = 'train',
    check_func=None
):
    inference_data = {key: read_jsonl(path) if os.path.exists(path) else None for key, path in jsonl_paths.items()}
    raw_data = {key: read_jsonl(path) if os.path.exists(path) else None for key, path in raw_paths.items()}
    os.makedirs(out_dir, exist_ok=True)
    result = {}
    incorrect_rows = []
    for key in inference_data.keys():
        if inference_data[key] is None or raw_data[key] is None:
            logger.warning(f'文件不存在, 跳过 ppl={key}测试')
            continue
        raw_d = {str(item['index']): item for item in raw_data[key]}
        n_incorrect = 0
        n_correct = 0
        n_toolong = 0
        for item in inference_data[key]:
            index = item['custom_id']
            raw_d_item = raw_d[index]
            gt_answer = parse_gt_answer(raw_d_item['names'], raw_d_item['solution'])
            # 解析推理结果
            message = item['response']['body']['choices'][0]['message']
            content, reasoning_content = message['content'], message['reasoning_content']
            response = '<think>' + reasoning_content + '</think>' + content # 是否调用reasoning content没有区别
            if check_func(response, gt_answer) == 1:
                n_correct += 1
            else:
                if item['response']['body']['usage']['completion_tokens'] == max_token_detection:
                    n_toolong += 1
                else:
                    n_incorrect += 1
                    incorrect_rows.append({
                        'index': index,
                        'gt_answer': gt_answer,
                        'response': content,
                        # 'reasoning_content': reasoning_content
                    })
        result[key] = {'pass@1': n_correct / len(inference_data[key]), 'n_correct': n_correct, 'n_total': len(inference_data[key])}
        logger.info(f"label={label}, key={key}, API调用推理正确: {n_correct}, API调用推理错误: {n_incorrect}, API调用推理超长: {n_toolong}")
    total_correct = sum([v['n_correct'] for v in result.values()])
    total_n = sum([v['n_total'] for v in result.values()])
    result['total'] = {'pass@1': total_correct / total_n, 'n_correct': total_correct, 'n_total': total_n}
    # 输出结果
    with open(f"{out_dir}/{label}_eval.json", 'w') as f:
        json.dump(result, f, indent=4)
    # 输出错误的行
    with open(f"{out_dir}/{label}_incorrect_examples.jsonl", 'w') as f:
        for item in incorrect_rows:
            f.write(json.dumps(item) + '\n')
    

        

        
def generate_data_for_RL(
    raw_jsonls: list[str],
    rl_output_dir: str,
    label: str,
    model: str='qwen'
):
    os.makedirs(rl_output_dir, exist_ok=True)
    logger.info(f"label={label}, 生成数据, raw_jsonls={raw_jsonls}")
    gen_end = ' Your final answer should be in this format: \\boxed{XX is a knave, YY is a knight, ...}'
    origin_head, origin_end, _ = get_tokenize_format(model)
    data = []
    for raw_jsonl_path in raw_jsonls:
        data += read_jsonl(raw_jsonl_path)
    result = []
    """
    {"answer":"34","gt_answer":"34","target":"34","data_source":"simplelr_qwen","ability":"math","prompt":[{"content":"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.<|im_end|>\n<|im_start|>assistant","role":"user"}],"reward_model":{"ground_truth":"34","style":"rule"},"extra_info":{"answer":"34","index":0,"level":1,"question":"","split":"train"}}
    """
    for d in data:
        d_input = origin_head + d['quiz'] + gen_end + origin_end
        gt_answer = parse_gt_answer(d['names'], d['solution'])
        level = len(gt_answer.split(', ')) # 有多少个人
        result.append({
            'prompt': [{'content': d_input, 'role': 'user'}],
            'answer': gt_answer,
            'gt_answer': gt_answer,
            'target': gt_answer,
            'data_source': 'kk',
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
    
def entry_process_raw_jsonl():
    process_raw_jsonl(
        raw_jsonl_paths=[
            'data/KK-Data/raw/test/people2_num200.jsonl',
            'data/KK-Data/raw/test/people3_num800.jsonl'
        ],
        preprocess_dir='data/KK-Data/preprocessed'
    )

def entry_generate_short_cot():
    generate_short_cot(
        prep_jsonl_path='data/KK-Data/preprocessed/prep_train.jsonl',
        sft_out_dir='data/KK-Data/sft_short_cot',
        label='train'
    )
    generate_short_cot(
        prep_jsonl_path='data/KK-Data/preprocessed/prep_valid.jsonl',
        sft_out_dir='data/KK-Data/sft_short_cot',
        label='val'
    )

def entry_generate_batch_inference():
    generate_batch_inference(
        prep_jsonl_path='data/KK-Data/preprocessed/prep_train.jsonl',
        preprocess_dir='data/KK-Data/preprocessed',
        label='train'
    )
    generate_batch_inference(
        prep_jsonl_path='data/KK-Data/preprocessed/prep_valid.jsonl',
        preprocess_dir='data/KK-Data/preprocessed',
        label='val'
    )

def entry_generate_long_cot_from_batch_inference():
    from verl.utils.reward_score.kk_no_format import compute_score_kk, comput_score_format
    generate_long_cot_from_batch_inference(
        inference_out_jsonl_path='data/KK-Data/preprocessed/batch_result_val.jsonl',
        prep_jsonl_path='data/KK-Data/preprocessed/prep_valid.jsonl',
        sft_out_dir='data/KK-Data/sft_long_cot',
        label='val',
        check_func=compute_score_kk
    )
    generate_long_cot_from_batch_inference(
        inference_out_jsonl_path='data/KK-Data/preprocessed/batch_result_train.jsonl',
        prep_jsonl_path='data/KK-Data/preprocessed/prep_train.jsonl',
        sft_out_dir='data/KK-Data/sft_long_cot',
        label='train',
        check_func=compute_score_kk
    )

def entry_generate_data_for_RL(model: str):
    assert model in ['qwen', 'base']
    # 为RL生成训练集和测试集
    generate_data_for_RL(
        raw_jsonls=[
            'data/KK-Data/raw/train/people3_num2000.jsonl',
            'data/KK-Data/raw/train/people4_num2000.jsonl',
            'data/KK-Data/raw/train/people5_num2000.jsonl',
            'data/KK-Data/raw/train/people6_num2000.jsonl',
            'data/KK-Data/raw/train/people7_num2000.jsonl',
            # ppl=8不参与训练, 只作为测试
        ],
        rl_output_dir=f'data/KK-Data/RL_data_{model}',
        label='train',
        model=model
    )
    generate_data_for_RL(
        raw_jsonls=[
            'data/KK-Data/raw/test/people4_num100.jsonl',
            'data/KK-Data/raw/test/people5_num100.jsonl',
            'data/KK-Data/raw/test/people6_num100.jsonl',
            'data/KK-Data/raw/test/people7_num100.jsonl',
            'data/KK-Data/raw/test/people8_num100.jsonl',
        ],
        rl_output_dir=f'data/KK-Data/RL_data_{model}',
        label='test',
        model=model
    )

def entry_generate_cot_from_actor(mode=['origin_sft', 'sample_from_buffer']):
    # 从actor生成结果中取出数据
    if mode == 'origin_sft':
        # checkpoint先生成SFT的解, 再整理成数据集
        out_dir = 'data/KK-Data/sft_actor_cot_1'
        generated_data = read_jsonl('data/KK-Data/actor_generated/long_3_step_125.jsonl')
        generated_data = [{'prompt': item['prompt'], 'response': item['responses'][0], 'gt_answer': item['gt_answer']} for item in generated_data]
    elif mode == 'sample_from_buffer':
        # 直接从long-3的replay buffer抽取正确结果
        out_dir = 'data/KK-Data/sft_actor_cot_2'
        generated_data = read_jsonl('outputs/verl-server-1-kk-long-3/step_125.jsonl')
        for i in [115, 116, 117, 118, 119, 120, 121, 122, 123, 124]:
            generated_data.extend(read_jsonl(f'outputs/verl-server-1-kk-long-3/step_{i}.jsonl'))
        
        generated_data = [
            {
                'prompt': item['output'].split('<|im_start|>assistant')[0] + '<|im_start|>assistant', 
                'response': item['output'].split('<|im_start|>assistant')[1], 
                'gt_answer': item['ground_truth'],
                'index': item['index']
            } for item in generated_data if item['correctness'] > 0
        ]
        valid_idx = []
        index_set = set()
        for id, item in enumerate(generated_data):
            if item['index'] not in index_set:
                valid_idx.append(id)
                index_set.add(item['index'])
        generated_data = [generated_data[id] for id in valid_idx]

        random.shuffle(generated_data)
        generated_data = generated_data[:1000]
    else:
        assert 0


    os.makedirs(out_dir, exist_ok=True)
        

    logger.warning("会使用long_cot_dir中的val.jsonl作为验证集, 事先检查格式是否正确")
    from verl.utils.reward_score.kk_no_format import compute_score
    # 读取文件
    result = []
    for item in generated_data:
        if compute_score(item['response'], item['gt_answer'])['correctness']:
            result.append({
                # 'id': int(item['custom_id']),
                'prompt': item['prompt'],
                'response': item['response'],
                'gt_answer': item['gt_answer']
            })
    logger.info(f"Valid samples: {len(result)}/{len(generated_data)}")
    # 保存数据到jsonl文件
    save_jsonl(f"{out_dir}/train.jsonl", result)
    # 保存数据为parquet文件
    convert2pq(f"{out_dir}/train.jsonl", f"{out_dir}/train.parquet")
    # 复制val文件
    import shutil
    long_cot_dir = 'data/KK-Data/sft_long_cot'
    shutil.copyfile(f"{long_cot_dir}/val.jsonl", f"{out_dir}/val.jsonl")
    shutil.copyfile(f"{long_cot_dir}/val.parquet", f"{out_dir}/val.parquet")

def entry_generate_batch_inference_for_eval(out_dir, model_name: str, label: str):
    # 为R1生成测试数据
    for n in [4,5,6,7,8,9,10]:
        generate_batch_inference_from_raw(
            raw_jsonl_path=f'data/KK-Data/raw/test/people{n}_num100.jsonl',
            out_dir=out_dir,
            label=f'test_{label}_{n}',
            model_name=model_name
        )

def entry_generate_final_eval_set(model=['qwen', 'r1_distill', 'raw', 'base'], use_hard=False):
    assert model in ['qwen', 'r1_distill', 'raw', 'base']
    
    # 为main_generation生成评估格式数据集
    if use_hard:
        out_dir = f'data/KK-Data/RL_{model}_eval_hard'
        os.makedirs(out_dir, exist_ok=True)
        pq_path = 'data/KK-Data/RL_data_hard/test.parquet'
    else:
        out_dir = f'data/KK-Data/RL_{model}_eval'
        os.makedirs(out_dir, exist_ok=True)
        pq_path = 'data/KK-Data/RL_data/test.parquet'
    data = pd.read_parquet(pq_path).to_dict(orient='records')
    result = []
    for item in data:
        prompt: str = item['prompt'][0]['content']
        gt_answer = item['gt_answer']
        if model == 'qwen':
            prompt = prompt
        elif model == 'raw':
            origin_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            origin_end = "<|im_end|>\n<|im_start|>assistant"
            prompt = prompt.removeprefix(origin_head).removesuffix(origin_end)
        elif model == 'base':
            origin_head = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            origin_end = "<|im_end|>\n<|im_start|>assistant"
            prompt = prompt.removeprefix(origin_head).removesuffix(origin_end)
            prompt = 'Question:\n' + prompt + "Your final answer should be in this format: \\boxed{XX is a knave, YY is a knight, ...}\nAnswer:\nLet's think step by step."
        else:
            assert 0
            
        result.append({
            'prompt': prompt,
            'data_source': item['data_source'],
            'extra_info': {
                'gt_answer': gt_answer,
                'index': item['extra_info']['index'],
                'level': item['extra_info']['level'],
            },
        })
    # 保存数据到jsonl文件和pq文件
    save_jsonl(f"{out_dir}/eval.jsonl", result)
    convert2pq(f"{out_dir}/eval.jsonl", f"{out_dir}/eval.parquet")

def entry_eval_baseline_answer(model_type: str, out_dir_name, max_token_detection=16384):
    # 从推理模型的batch inference中计算pass@1
    from verl.utils.reward_score.kk_no_format import compute_score_kk
    eval_baseline(
        jsonl_paths={
            4: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_4.jsonl',
            5: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_5.jsonl',
            6: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_6.jsonl',
            7: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_7.jsonl',
            8: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_8.jsonl',
            9: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_9.jsonl',
            10: f'data/KK-Data/{model_type}_eval/test_kk_{model_type}_10.jsonl',
        },
        raw_paths={
            4: 'data/KK-Data/raw/test/people4_num100.jsonl',
            5: 'data/KK-Data/raw/test/people5_num100.jsonl',
            6: 'data/KK-Data/raw/test/people6_num100.jsonl',
            7: 'data/KK-Data/raw/test/people7_num100.jsonl',
            8: 'data/KK-Data/raw/test/people8_num100.jsonl',
            9: 'data/KK-Data/raw/test/people9_num100.jsonl',
            10:'data/KK-Data/raw/test/people10_num100.jsonl',
        },
        out_dir=f'outputs/eval/{out_dir_name}',
        max_token_detection=max_token_detection,
        label='test_4to10',
        check_func=compute_score_kk
    )

def entry_eval_online_baseline_performance():
    # 评估entry_generate_final_eval_set生成的结果的性能好坏
    import sys
    from verl.utils.reward_score.kk_no_format import compute_score_kk
    import numpy as np

    in_dir = 'data/KK-Data/online_generated'
    jsonls = [
        'ds_distill_qwen_7B.jsonl',
        'ds_distill_qwen_14B.jsonl',
        'ds_distill_qwen_32B.jsonl',
        'ds_v3_0324.jsonl',
        'qwen2.5_7b.jsonl',
        'qwen2.5_32b.jsonl'

    ]
    results = []
    for jsonl_name in jsonls:
        level_accs = {}
        jsonl_path = os.path.join(in_dir, jsonl_name)
        n_invalid = 0
        with jsonlines.open(jsonl_path, 'r') as f:
            for obj in f:
                response = obj['api_response']
                   
                level = obj['extra_info']['level']
                gt_ans = obj['extra_info']['gt_answer']
                if level not in level_accs:
                    level_accs[level] = []
                if response is None:
                    n_invalid += 1
                    correctness = 0.0
                else:
                    correctness = compute_score_kk(response, gt_ans)
                level_accs[level].append(correctness)
            level_accs = {level: np.asarray(v).mean() for level, v in level_accs.items()}
        top_8_avg_acc = np.mean([level_accs[level] for level in range(4,9)])
        top_10_avg_acc = np.mean([level_accs[level] for level in range(4,11)])
        results.append(f'\njsonl_name: {jsonl_name}, level_accs={sorted(level_accs.items(), key=lambda x: x[0])}, invalid={n_invalid}, 8_avg={top_8_avg_acc}, 10_avg={top_10_avg_acc}')
        logger.info(results[-1])
        
    with open(os.path.join(in_dir, 'eval_result.log'), 'w', encoding='utf-8') as f:
        f.writelines(results)


if __name__ == '__main__':
    import os

    # entry_process_raw_jsonl()
    # entry_generate_short_cot()

    # Generate batch inference data -> remote Deepseek-R1 inference -> collect inference result
    # entry_generate_batch_inference()
    # entry_generate_long_cot_from_batch_inference()

    # Generate RL dataset
    # entry_generate_data_for_RL(model='base')

    """Evaluate local model"""
    # entry_generate_final_eval_set(model='qwen', use_hard=False)
    # entry_generate_final_eval_set(model='base', use_hard=False)

    
    """Evaluate deepseek R1"""
    #entry_generate_batch_inference_for_eval(model_name='deepseek-ai/DeepSeek-R1', out_dir='data/KK-Data/r1_eval', label='R1')
    #entry_eval_baseline_answer(model_type='r1', out_dir_name='kk_deepseek_r1_eval', max_token_detection=16384)

    """Evaluate QwQ"""
    # entry_generate_batch_inference_for_eval(model_name='Qwen/QwQ-32B', out_dir='data/KK-Data/qwq_eval', label='qwq')
    #entry_eval_baseline_answer(model_type='qwq', out_dir_name='kk_deepseek_qwq_eval', max_token_detection=16384)

    """Evaluate other online models"""
    # entry_generate_final_eval_set(model='raw', use_hard=True)
    # entry_eval_online_baseline_performance()



