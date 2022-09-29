import json
from tqdm import tqdm
from functools import cmp_to_key
import math


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
        answer = tuple([item.strip() for item in str(answer_str).split(", ")])
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
    with open("../results_tab_fact_500/unfiltered_tab_fact_qa.json", "r") as f:
        all_data = json.load(f)
    right = 0
    # map = {"entailed": "1", "refuted": "0"}

    for eid, data_item in tqdm(all_data.items()):
        try:
            pred = majority_vote(data_item['generations']['default_template'])
        except:
            pred = []
        gold = data_item['ori_data_item']['label']
        if pred[0] == str(gold):
            right += 1

    print("Acc: {}/{} ".format(right, len(all_data)), right / len(all_data))
