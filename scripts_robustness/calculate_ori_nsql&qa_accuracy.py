import json
import argparse
import os

from utils.utils import load_data_split, eval_ex_match, majority_vote


def main():
    dataset = load_data_split(args.dataset, args.dataset_split)
    full_dataset = load_data_split('wikitq', args.dataset_split)
    wtq_id2full_dataset_id = {full_dataset[idx]['id']: idx for idx in range(len(full_dataset))}
    # Calculate nsql accuracy
    with open(os.path.join(args.ori_result_dir, args.ori_nsql_exec_file), 'r') as f:
        nsql_result_dict = json.load(f)
    with open(os.path.join(args.new_result_dir, args.new_nsql_exec_file), 'r') as f:
        new_nsql_result_dict = json.load(f)
    n_correct_nsql, n_correct_new_nsql = 0, 0
    for idx, data_item in enumerate(dataset):
        full_dataset_id = wtq_id2full_dataset_id.get(data_item['id'], -1)
        if full_dataset_id == -1:
            continue
        result_item = nsql_result_dict[str(full_dataset_id)]
        new_result_item = new_nsql_result_dict[str(idx)]
        score = eval_ex_match(
            result_item['pred_answer'],
            result_item['gold_answer'],
            level=4,
            question=data_item['question']
        )
        score_new = eval_ex_match(
            new_result_item['pred_answer'],
            new_result_item['gold_answer'],
            level=4,
            question=data_item['question']
        )
        # if score != score_new:
        #     print('+' * 80)
        #     print(data_item['id'])
        #     print('old pred: ', result_item['pred_answer'])
        #     print('new pred: ', new_result_item['pred_answer'])
        #     print('gold: ', result_item['gold_answer'])
        n_correct_nsql += score
        n_correct_new_nsql += score_new
    print(f"NSQL accuracy: {n_correct_nsql}/{len(dataset)}={n_correct_nsql/len(dataset)}")
    print(f"NSQL new accuracy: {n_correct_new_nsql}/{len(dataset)}={n_correct_new_nsql/len(dataset)}")
    # Calculate qa accuracy
    with open(os.path.join(args.ori_result_dir, args.ori_qa_file), 'r') as f:
        qa_result_dict = json.load(f)
    with open(os.path.join(args.new_result_dir, args.new_qa_file), 'r') as f:
        new_qa_result_dict = json.load(f)
    n_correct_qa, n_correct_new_qa = 0, 0
    for idx, data_item in enumerate(dataset):
        full_dataset_id = wtq_id2full_dataset_id.get(data_item['id'], -1)
        if full_dataset_id == -1:
            continue
        result_item = qa_result_dict[str(full_dataset_id)]
        new_result_item = new_qa_result_dict[str(idx)]
        try:
            qa_answer_list = [[g[0]] for g in result_item['generations']['default_template']]
            pred_answer, pred_answer_nsqls = majority_vote(
                nsqls=result_item['generations']['default_template'],
                pred_answer_list=qa_answer_list,
                allow_none_and_empty_answer=False,
                answer_placeholder='<error|empty>',
                vote_method='simple',
            )
        except:
            pred_answer = []
        try:
            new_qa_answer_list = [[g[0]] for g in new_result_item['generations']['default_template']]
            new_pred_answer, new_pred_answer_nsqls = majority_vote(
                nsqls=result_item['generations']['default_template'],
                pred_answer_list=new_qa_answer_list,
                allow_none_and_empty_answer=False,
                answer_placeholder='<error|empty>',
                vote_method='simple',
            )
        except:
            new_pred_answer = []
        question = data_item['question']
        score = eval_ex_match(pred_answer, result_item['ori_data_item']['answer_text'], level=4, question=question)
        score_new = eval_ex_match(new_pred_answer, new_result_item['ori_data_item']['answer_text'], level=4, question=question)
        # if (not score) and score_new:
        #     print(score, score_new)
        #     print('-' * 80)
        #     print(data_item['id'])
        #     print('old pred: ', pred_answer)
        #     print('new pred: ', new_pred_answer)
        #     print('gold: ', gold_answer)
        n_correct_qa += score
        n_correct_new_qa += score_new
    print(f"QA accuracy: {n_correct_qa}/{len(dataset)}={n_correct_qa/len(dataset)}")
    print(f"QA new accuracy: {n_correct_new_qa}/{len(dataset)}={n_correct_new_qa/len(dataset)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitq_robustness',
                        choices=['wikitq_robustness'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    # file path or name
    parser.add_argument('--ori_result_dir', type=str, default='../results_wikitq_full/')
    parser.add_argument('--ori_nsql_exec_file', type=str, default='dev_wikitq_unfiltered_executions_prompt_v3_no_CoT_new_exec_8shotqa.json')
    parser.add_argument('--ori_qa_file', type=str, default='dev_unfiltered_wikitq_qa_balanced_enter_fixed.json')
    parser.add_argument('--new_result_dir', type=str, default='../results_robustness/')
    parser.add_argument('--new_nsql_exec_file', type=str, default='unfiltered_wikitq_robustness_nsql_executions.json')
    parser.add_argument('--new_qa_file', type=str, default='unfiltered_wikitq_robustness_qa.json')


    args = parser.parse_args()
    main()