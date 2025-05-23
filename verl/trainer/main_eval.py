# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import math, gsm8k, hf_math_verify, kk_no_format
import pandas as pd
import numpy as np


def select_reward_fn(data_source):
    if data_source in ['math_length', 'math_base']:
        return hf_math_verify.compute_score_with_length
    elif data_source == 'kk':
        return kk_no_format.compute_score
    else:
        raise NotImplementedError

def compute_level_metrics(batch: list[dict]):
    from collections import defaultdict
    metrics = {}
    # 统计train/test set中不同level的正确率
    level_correctness = defaultdict(list)
    level_clip_ratio = defaultdict(list)
    #level_response_lengths = defaultdict(list)
    eos_tokens = ['<|im_end|>', '<|endoftext|>', '<|finetune_right_pad_id|>', '<|eot_id|>', '<｜end▁of▁sentence｜>']
    for i in range(len(batch)):
        # 还在math_length上应用了
        # assert batch[i].non_tensor_batch['data_source'] == 'kk'
        
        level = batch[i]['extra_info']['level']
        correctness = int(batch[i]['avg_score'].item())
        level_correctness[level].append(correctness)
        # 计算每个level的clip_ratio
        # print(self.tokenizer.convert_ids_to_tokens(batch[i].batch['responses'][-1].item()))
        is_clipped = int(not any([et in batch[i]['responses'][0] for et in eos_tokens]))
        level_clip_ratio[level].append(is_clipped)

    # 计算每个level的正确率和clip_ratio
    for level, correctness_list in level_correctness.items():
        correctness = np.mean(correctness_list)
        metrics[f'level_correctness/{level}_correctness'] = correctness
    for level, clip_ratio_list in level_clip_ratio.items():
        clip_ratio = np.mean(clip_ratio_list)
        metrics[f'level_clip_ratio/{level}_clip_ratio'] = clip_ratio
    return metrics


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_local_path_from_hdfs(config.data.path)
    if local_path.endswith('.parquet'):
        dataset = pd.read_parquet(local_path).to_dict(orient='records')
    elif local_path.endswith('.jsonl'):
        dataset = pd.read_json(local_path, lines=True).to_dict(orient='records')
    # prompts = dataset[config.data.prompt_key]
    #responses = dataset[config.data.response_key]
    #data_sources = dataset[config.data.data_source_key]
    # reward_model_data = dataset[config.data.reward_model_key]

    passes = 0

    for i, item in enumerate(dataset):
        response_lst = item[config.data.response_key]
        if isinstance(response_lst, str):
            response_lst = [response_lst]
        
        data_source = item[config.data.data_source_key]
        # select reward score based on data_source
        reward_data = item[config.data.reward_model_key]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['gt_answer']
        score_lst = []
        for r in response_lst:
            if data_source in ['math_length', 'kk', 'math_base']: # 对于base模型来说是必要的
                r += '<|endoftext|>'
            score = float(reward_fn(r, ground_truth)['correctness'])
            score_lst.append(score)
        score_lst = np.asarray(score_lst)
        dataset[i]['avg_score'] = np.mean(score_lst)
        passes += dataset[i]['avg_score']
            
    
    print(f'pass@1(avg): {passes / len(dataset)}')
    print('metrics:')
    print(compute_level_metrics(dataset))
    import os, datetime
    with open(os.path.join(config.data.out_dir, 'eval_result.log'), 'w', encoding='utf-8') as f:
        f.write(str(datetime.datetime.now()) + '\n')
        f.write(str(config) + '\n')
        f.writelines([
            f'n_total={len(dataset)}\n',
            f'pass@1(avg): {passes / len(dataset)}\n',
            'metrics:\n',
            str(compute_level_metrics(dataset))
        ])


if __name__ == '__main__':
    main()
