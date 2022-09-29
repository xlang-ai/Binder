import json
from tqdm import tqdm

from utils.utils import eval_ex_match, majority_vote, load_data_split


if __name__ == '__main__':
    # Load data
    missing_squall_dataset = load_data_split('missing_squall', 'validation')
    wikitq_sql_unsolvable_but_in_squall_dataset = \
        load_data_split('wikitq_sql_unsolvable_but_in_squall', 'validation')
    wikitq_sql_solvable_dataset = load_data_split('wikitq_sql_solvable', 'validation')
    wikitq_dataset = load_data_split('wikitq', 'validation')
    missing_squall_ids = set(item['id'] for item in missing_squall_dataset)
    wikitq_sql_unsolvable_but_in_squall_ids = set(item['id'] for item in wikitq_sql_unsolvable_but_in_squall_dataset)
    wikitq_sql_solvable_ids = set(item['id'] for item in wikitq_sql_solvable_dataset)
    missing_squall_wtqid_to_id = \
        {missing_squall_dataset[idx]['id']: idx
         for idx in range(len(missing_squall_dataset))}
    wikitq_sql_unsolvable_but_in_squall_wtqid_to_id = \
        {wikitq_sql_unsolvable_but_in_squall_dataset[idx]['id']: idx
         for idx in range(len(wikitq_sql_unsolvable_but_in_squall_dataset))}
    wikitq_sql_solvable_wtqid_to_id = \
        {wikitq_sql_solvable_dataset[idx]['id']: idx
         for idx in range(len(wikitq_sql_solvable_ids))}

    with open("../results_internal_knowledge/dev_unfiltered_wikitq_qa.json", "r") as f:
        all_data = json.load(f)

    # Calculate accuracy respectively
    n_correct_missing_squall, n_correct_wikitq_sql_unsolvable_but_in_squall, n_correct_wikitq_sql_solvable = 0, 0, 0
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
            # pass
            print(question)
            print(pred_answer, gold_answer)
            print()
        n_correct += score

    print("Accuracy: {}/{} ".format(n_correct, len(all_data)), n_correct / len(all_data))