from K_and_K_preprocess import read_jsonl, save_jsonl, convert2pq
import random, os
from loguru import logger
from os.path import join as osjoin
def entry_generate_batch_inference_from_actor():
    # 从replay buffer里面找2K数据, 用deepseek蒸镏至少1K数据, 目的是证明光靠SFT无法有效传递信息
    pass

def entry_generate_cot_from_actor(mode=['origin_sft', 'sample_from_buffer', 'ds_batch_inference']):
    # 从actor生成结果中取出数据
    # 注意: sample_from_buffer没有添加seed, 这个脚本跑不出之前的结果
    # ds_batch_inference: 为ds R1生成结果
    if mode == 'origin_sft':
        # checkpoint先生成SFT的解, 再整理成数据集
        out_dir = 'data/KK-Data/sft_actor_cot_1'
        generated_data = read_jsonl('data/KK-Data/actor_generated/long_3_step_125.jsonl')
        generated_data = [{'prompt': item['prompt'], 'response': item['responses'][0], 'gt_answer': item['gt_answer']} for item in generated_data]
    elif mode  == 'sample_from_buffer':
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
            } for item in generated_data if (item['correctness'])
        ]
        valid_idx = []
        index_set = set()
        for id, item in enumerate(generated_data):
            if item['index'] not in index_set:
                valid_idx.append(id)
                index_set.add(item['index'])
        generated_data = [generated_data[id] for id in valid_idx]

        random.seed(0)
        random.shuffle(generated_data)
        if mode == 'sample_from_buffer':
            generated_data = generated_data[:1000]
        else:
            generated_data = generated_data[:763]
    elif mode == 'ds_batch_inference':
        out_dir = 'data/KK-Data/preprocessed'
        generated_data = read_jsonl('outputs/verl-server-1-kk-long-3/step_125.jsonl')
        for i in [115, 116, 117, 118, 119, 120, 121, 122, 123, 124]:
            generated_data.extend(read_jsonl(f'outputs/verl-server-1-kk-long-3/step_{i}.jsonl'))
        valid_idx = []
        index_set = set()
        for id, item in enumerate(generated_data):
            if item['index'] not in index_set:
                valid_idx.append(id)
                index_set.add(item['index'])
        generated_data = [generated_data[id]for id in valid_idx]
        logger.info(f"为RL准备{len(generated_data)}个独立样本")
        
        d_sample = {
            "custom_id": "request-1", 
            "method": "POST", 
            "url": "/v1/chat/completions",
            "body": {
                "model": 'deepseek-ai/DeepSeek-R1', 
                "messages": [{"role": "user", "content": "How does photosynthesis work?"}], 
                "stream": True, 
                "max_tokens": 16384,
                "temperature": 0.0,
                # "top_p": 0.95,
        }}
        from copy import deepcopy
        
        result = []
        prepared_data = []
        origin_head = '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n'
        origin_end = '<|im_end|>\n<|im_start|>assistant'
        for b in generated_data:
            prompt = b['output'].split(origin_end)[0].removeprefix(origin_head)
            d_sample['body']['messages'][0]['content'] = prompt
            d_sample['custom_id'] =  str(b['index'])
            result.append(deepcopy(d_sample))
            prepared_data.append({
                'quiz': prompt,
                'index': str(b['index']),
                'ground_truth': b['ground_truth'],
                'data_source': b['data_source'],
                'level': b['level']
            })
        save_jsonl(osjoin(out_dir, 'actor_2_ds_R1_prepared.jsonl'), prepared_data)
        save_jsonl(osjoin(out_dir, 'actor_2_ds_R1_inference.jsonl'), result)
        return
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


def generate_long_cot_from_batch_inference(
    inference_out_jsonl_path: str,
    prep_jsonl_path: str,
    sft_out_dir: str,
    check_func=None
):
    # 从批量推理结果生成长cot
    os.makedirs(sft_out_dir, exist_ok=True)
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
        gt_answer = prep_d['ground_truth']
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
        
    logger.info(f"API调用推理正确: {n_correct}, API调用推理错误: {n_incorrect}, API调用推理超长: {n_toolong}")
    import numpy as np
    # 打乱顺序
    random.seed(1234)
    random.shuffle(result)
    # 重新设置id
    for i, item in enumerate(result):
        item['id'] = i
    
    for percent in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]:
        logger.info(f"总长度{percent}%分位数: {np.percentile(total_tokens_list, percent)}")


    train_result = result[:1000]
    val_result = result[-50:]
    logger.info(f'总样本量: {len(result)}, train: {len(train_result)}, valid: {len(val_result)}')
    # 保存数据到jsonl文件
    save_jsonl(f"{sft_out_dir}/train.jsonl", train_result)
    convert2pq(f"{sft_out_dir}/train.jsonl", f"{sft_out_dir}/train.parquet")
    save_jsonl(f"{sft_out_dir}/val.jsonl", val_result)
    convert2pq(f"{sft_out_dir}/val.jsonl", f"{sft_out_dir}/val.parquet")



def entry_generate_actor2_sft_from_r1_response():
    from verl.utils.reward_score.kk_no_format import compute_score_kk

    generate_long_cot_from_batch_inference(
        inference_out_jsonl_path='data/KK-Data/preprocessed/actor_2_ds_R1_result.jsonl',
        prep_jsonl_path='data/KK-Data/preprocessed/actor_2_ds_R1_prepared.jsonl',
        sft_out_dir='data/KK-Data/actor_2_r1_sft',
        check_func=compute_score_kk
    )

if __name__ == '__main__':
    # collect re-distillation data from replay buffer (re-distill-rl-kk) 
    # entry_generate_cot_from_actor(mode='sample_from_buffer')

    # collect re-distillation data from SFT data (re-distill-sft-kk)
    # entry_generate_cot_from_actor(mode='origin_sft')

    # collect questions for DeepSeek-R1 inference
    # entry_generate_cot_from_actor(mode='ds_batch_inference')
    # generate SFT dataset from DeepSeek-R1 inference result
    entry_generate_actor2_sft_from_r1_response()


