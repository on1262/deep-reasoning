
import re

def comput_score_format(solution_str: str, ground_truth: str):
    # 该函数只检查boxed内格式是否正确, 不检查内容
    gt_answers = ', '.join(sorted(ground_truth.split(', ')))
    gt_answers = gt_answers.replace('knight', 'knave')
    # 提取boxed
    boxed_pattern = re.compile(r'\\boxed{(.*?)}')
    boxed_match = boxed_pattern.findall(solution_str)
    if len(boxed_match) > 0:
        # 检查最后一个匹配项
        boxed_content = boxed_match[-1]
        boxed_content = boxed_content.replace('knight', 'knave')
        # 只匹配人名
        out_answers = ', '.join(sorted(boxed_content.split(', ')))
        return 1.0 if out_answers == gt_answers else 0.0
    else:
        return 0.0


def detect_duplicate_output(s: str, min_duplicate_count=10) -> bool:
    # 检查是否有重复输出
    # 使用正则表达式查找重复出现大于等于N次的子串
    pattern = re.compile(r'(.+?)\1{DUP_COUNT,}'.replace('DUP_COUNT', str(min_duplicate_count - 1)), re.DOTALL)
    if len(s) < 1000:
        return bool(pattern.search(s))
    else:
        s1 = s[-500:]
        s2 = s[len(s)//2-250:len(s)//2+250]
        return bool(pattern.search(s1)) or bool(pattern.search(s2))

def compute_score_kk(solution_str: str, ground_truth: str):
    # solution: .... \\boxed{A is knight, B is knave}
    # ground_truth: A is knight, B is knave
    gt_answers = ', '.join(sorted(ground_truth.split(', ')))
    # 提取boxed
    boxed_pattern = re.compile(r"\\boxed\{([^{}]*)\}")
    boxed_match = boxed_pattern.findall(solution_str)
    if len(boxed_match) > 0:
        # 检查最后一个匹配项
        boxed_content = boxed_match[-1]
        # boxed_content = boxed_match.group(1)
        # 严格匹配
        out_answers = ', '.join(sorted(boxed_content.split(', ')))
        return 1.0 if out_answers == gt_answers else 0.0

    else:
        return 0.0

def compute_score(solution_str: str, ground_truth: str):
    """
    标准:
    1. 存在<|endoftext|>标记, 否则一律为0
    3. 答案不匹配时一律为0
    """
    if (not solution_str.endswith("<|endoftext|>")) and (not solution_str.endswith("<|im_end|>")):
        return {"score": 0.0, "correctness": False}
    else:
        score = compute_score_kk(solution_str, ground_truth)
        return {"score": score, "correctness": score > 0.5}


def unit_test_kk():
    # 测试compute_score_kk函数
    solution_str = "The final answer is: \\boxed{A is knight, B is knave}."
    ground_truth = "A is knight, B is knave"
    assert compute_score_kk(solution_str, ground_truth) == 1

    solution_str = "The final answer is: \\boxed{A is knight, A is knave}\\boxed{A is knight, B is knave}."
    ground_truth = "A is knight, B is knave"
    assert compute_score_kk(solution_str, ground_truth) == 1

    solution_str = "The final answer is: \\boxed{B is knave, A is knight}."
    ground_truth = "B is knave, A is knight"
    assert compute_score_kk(solution_str, ground_truth) == 1

    solution_str = "The final answer is: \\boxed{A is knight, A is knave, B is knave, B is knave}."
    ground_truth = "A is knight, B is knave"
    assert compute_score_kk(solution_str, ground_truth) == 0

    solution_str = "The final answer is: \\boxed{}."
    ground_truth = "A is knight, B is knave"
    assert compute_score_kk(solution_str, ground_truth) == 0

    solution_str = "The final answer is: A is knight, A is knave."
    ground_truth = "A is knight, B is knave"
    assert compute_score_kk(solution_str, ground_truth) == 0

    solution_str = "\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight, Lily is a knave, Penelope is a knave}\n\\boxed{Avery is a knight"
    ground_truth = "Avery is a knight, Lily is a knave, Penelope is a knave"
    assert compute_score_kk(solution_str, ground_truth) == 0

if __name__ == '__main__':
    unit_test_kk()