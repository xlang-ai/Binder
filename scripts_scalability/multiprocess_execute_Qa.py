import json
from tqdm import tqdm
from functools import cmp_to_key
import math

from utils.utils import eval_ex_match, majority_vote, load_data_split

if __name__ == '__main__':
    with open("../results_scalability/unfiltered_wikitq_scalability_500rows_qa.json", "r") as f:
        all_data = json.load(f)

    # Calculate accuracy respectively
    n_correct = 0
    for eid, data_item in tqdm(all_data.items()):
        wtq_id = data_item['ori_data_item']['id']
        qa_answer_list = [[g[0]] for g in data_item['generations']['default_template']]
        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=data_item['generations']['default_template'],
            pred_answer_list=qa_answer_list,
            allow_none_and_empty_answer=False,
            answer_placeholder='<error|empty>',
            vote_method='simple',
        )
        gold_answer = data_item['ori_data_item']['answer_text']
        question = data_item['ori_data_item']['question']
        nt_id = data_item['ori_data_item']['id']

        score = eval_ex_match(pred_answer, gold_answer, 4, question)
        if score:
            pass
            # print(question)
            # print(pred_answer, gold_answer)
            # print()
        n_correct += score

    print("Accuracy: {}/{} ".format(n_correct, len(all_data)), n_correct / len(all_data))
