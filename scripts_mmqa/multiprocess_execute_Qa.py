import json
from tqdm import tqdm
from functools import cmp_to_key
import math
import copy
from utils.utils import eval_ex_match


def majority_vote(answers, vote_with_weight=True):
    """
    Determine the final nsql execution answer by majority vote.
    """

    def _compare_answer_vote_with_count(a, b):
        """
        compare count sum.
        """
        return 1 if a[1]['count'] > b[1]['count'] else -1

    def _compare_answer_vote_with_weight(a, b):
        """
        compare prob sum.
        """
        return 1 if a[1]['prob'] > b[1]['prob'] else -1

    candi_answer_dict = {}

    for answer_str, _, logprob in answers:
        answer = tuple(str(answer_str).split(", "))
        if answer not in candi_answer_dict.keys():
            candi_answer_dict[answer] = {}
            candi_answer_dict[answer]['count'] = 0
            candi_answer_dict[answer]['prob'] = 0
        candi_answer_dict[answer]['count'] += 1
        candi_answer_dict[answer]['prob'] += math.exp(logprob)

    if vote_with_weight:
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_count),
                                          reverse=True)
    else:
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_weight),
                                          reverse=True)

    pred_answer_info = sorted_candi_answer_list[0]
    pred_answer = list(pred_answer_info[0])
    return pred_answer


if __name__ == '__main__':
    with open("../results_mmqa/unfiltered_mmqa_nsqls_mmqa_v2_standard_Qa.json", "r") as f:
        all_data = json.load(f)

    right = 0
    all_data_with_predictions = {}

    for eid, data_item in tqdm(all_data.items()):
        try:
            pred = majority_vote(data_item['generations']['default_template'])
        except:
            pred = []
        gold = data_item['ori_data_item']['answer_text'].split(" | ")
        question = data_item['ori_data_item']['question']

        new_data_item = {}
        new_data_item['pred_answer'] = pred
        new_data_item['gold_answer'] = gold
        new_data_item['question'] = question
        new_data_item['nsqls'] = data_item['generations']['default_template']

        all_data_with_predictions[eid] = new_data_item

    with open("../results_mmqa/unfiltered_mmqa_nsqls_mmqa_v2_standard_Qa_exec.json", "w") as f:
        json.dump(all_data_with_predictions, f, indent=4)

    with open("../results_mmqa/unfiltered_mmqa_nsqls_mmqa_v2_standard_Qa_exec.json", "r") as f:
        all_data_with_predictions = json.load(f)

    right = 0

    for eid, data_item in tqdm(all_data_with_predictions.items()):
        right += eval_ex_match(data_item['pred_answer'], data_item['gold_answer'], level=4, question=data_item['question'])
    print("acc", right/len(all_data_with_predictions.keys()))