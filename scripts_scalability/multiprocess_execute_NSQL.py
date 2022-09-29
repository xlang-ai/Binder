"""
Execute NSQLs with Codex as QA module.
"""

import time
import json
import argparse
from typing import List, Dict
from functools import cmp_to_key
import multiprocessing
import os
import time
import math

from nsql.nsql_exec import Executor, NeuralDB
from utils.normalizer import normalize_sql
from utils.utils import load_data_split, eval_ex_match, majority_vote


def worker_execute(
        pid,
        args,
        dataset,
        nsql_dict
):
    result_dict = dict()
    n_total_samples, n_correct_samples = 0, 0
    for eid, data_item in enumerate(dataset):
        eid = str(eid)
        if eid not in nsql_dict:
            continue
        print(f"Process#{pid}: eid {eid}, wtq-id {data_item['id']}")
        result_dict[eid] = dict()
        result_dict[eid]['question'] = data_item['question']
        result_dict[eid]['gold_answer'] = data_item['answer_text']
        n_total_samples += 1
        # Load table
        table = data_item['table']
        title = table['page_title']
        executor = Executor()
        # Execute
        tid = f"{args.dataset}_{args.dataset_split}_{data_item['id']}"
        exec_answer_list = []
        nsql_exec_answer_dict = dict()
        for idx, (nsql, _, logprob) in enumerate(nsql_dict[eid]['nsqls']):
            print(f"Process#{pid}: eid {eid}, tid {tid}, executing nsql#{idx}, logprob={logprob}")
            try:
                if nsql in nsql_exec_answer_dict:
                    exec_answer = nsql_exec_answer_dict[nsql]
                else:
                    db = NeuralDB([{"title": title, "table": table}], eid=int(eid))
                    nsql = normalize_sql(
                        sql_str=nsql,
                        all_headers=db.get_table_df().columns.to_list(),
                        table_title=title
                    )
                    exec_answer = executor.nsql_exec(nsql, db, verbose=args.verbose)
                    if isinstance(exec_answer, str):
                        exec_answer = [exec_answer]
                    nsql_exec_answer_dict[nsql] = exec_answer
                exec_answer_list.append(exec_answer)
            except Exception as e:
                print(f"Process#{pid}: Execution error {e}")
                exec_answer = '<error>'
                exec_answer_list.append(exec_answer)
            # Store tmp execution answers
            if nsql_dict[eid].get('exec_answers', None) is None:
                nsql_dict[eid]['exec_answers'] = []
            nsql_dict[eid]['exec_answers'].append(exec_answer)
        # Majority vote to determine the final prediction answer
        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=nsql_dict[eid]['nsqls'],
            pred_answer_list=exec_answer_list,
            allow_none_and_empty_answer=args.allow_none_and_empty_answer,
            answer_placeholder=args.answer_placeholder,
            vote_method=args.vote_method,
            answer_biased=args.answer_biased,
            answer_biased_weight=args.answer_biased_weight
        )
        # Evaluate
        result_dict[eid]['pred_answer'] = pred_answer
        result_dict[eid]['nsql'] = pred_answer_nsqls
        gold_answer = data_item['answer_text']
        score = eval_ex_match(pred_answer, gold_answer, level=args.eval_level, question=result_dict[eid]['question'])
        n_correct_samples += score
        print(f'Process#{pid}: pred answer: {pred_answer}')
        print(f'Process#{pid}: gold answer: {gold_answer}')
        if score == 1:
            print(f'Process#{pid}: Correct!')
        else:
            print(f'Process#{pid}: Wrong.')
        print(f'Process#{pid}: Accuracy: {n_correct_samples}/{n_total_samples}')

    # Save tmp execution answers
    with open(os.path.join(args.save_dir, f"tmp_nsql_dict_{pid}.json"), 'w') as f:
        json.dump(nsql_dict, f, indent=4)

    return result_dict


def main():
    # load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    # load nsqls and process as a unified format
    if args.exec_task == 'benchmark':
        with open(os.path.join(args.template_dir, args.input_nsql_file), 'r') as f:
            nsql_dict = json.load(f)['default_template']
            for eid, data_dict in nsql_dict.items():
                data_dict['nsqls'] = [(data_dict['nsql'], None, 0.)]  # 0. is pseudo logprob
                del data_dict['nsql']
    elif args.exec_task == 'full_validation':
        with open(os.path.join(args.save_dir, args.input_nsql_file), 'r') as f:
            data = json.load(f)
        nsql_dict = dict()
        for eid, data_dict in data.items():
            if data[eid]['generations'] and data[eid]['generations']['default_template']:
                nsqls = data[eid]['generations']['default_template']
            else:
                nsqls = [['<dummy nsql>', None, 0.]]
            nsql_dict[eid] = {'nsqls': nsqls}
    else:
        raise ValueError

    # split by processes
    nsql_dict_group = [dict() for _ in range(args.n_processes)]
    for idx, eid in enumerate(nsql_dict.keys()):
        if args.debug_id != -1 and int(eid) != args.debug_id:
            continue
        nsql_dict_group[idx % args.n_processes][eid] = nsql_dict[eid]

    # execute nsqls
    result_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        worker_results.append(pool.apply_async(worker_execute, args=(
            pid,
            args,
            dataset,
            nsql_dict_group[pid]
        )))
    # merge worker results
    for r in worker_results:
        worker_result_dict = r.get()
        result_dict.update(worker_result_dict)
    pool.close()
    pool.join()
    n_correct_samples = 0
    for eid, item in result_dict.items():
        pred_answer, gold_answer = item['pred_answer'], item['gold_answer']
        n_correct_samples += eval_ex_match(pred_answer, gold_answer,
                                           level=args.eval_level,
                                           question=result_dict[eid]['question'])
    print(f'Overall Accuracy: {n_correct_samples}/{len(result_dict)}')

    # save
    with open(os.path.join(args.save_dir, args.output_nsql_execution_file), 'w') as f:
        json.dump(result_dict, f)

    print(f'Done. Elapsed time: {time.time() - start_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import platform, multiprocessing

    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser.add_argument('--dataset', type=str, default='wikitq_scalability_ori',
                        choices=['wikitq_scalability_ori',
                                 'wikitq_scalability_100rows',
                                 'wikitq_scalability_200rows',
                                 'wikitq_scalability_500rows'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--template_dir', type=str, default='../templates/')
    parser.add_argument('--save_dir', type=str, default='../results_scalability')
    parser.add_argument('--api_keys_file', type=str, default='../keys.txt')
    parser.add_argument('--input_nsql_file', type=str,
                        default='unfiltered_wikitq_scalability_500rows_nsqls.json')
    parser.add_argument('--output_nsql_execution_file', type=str,
                        default='unfiltered_wikitq_scalability_500rows_nsql_executions.json')
    # multiprocess options
    parser.add_argument('--n_processes', type=str, default=4)
    # execute options
    parser.add_argument('--exec_task', type=str, default='full_validation',
                        choices=['benchmark', 'full_validation'])
    parser.add_argument('--use_majority_vote', action='store_false',
                        help='Whether use majority vote to determine the prediction answer.')
    parser.add_argument('--allow_none_and_empty_answer', action='store_true',
                        help='Whether regarding none and empty executions as a valid answer.')
    parser.add_argument('--answer_placeholder', type=str, default='<error|empty>',
                        help='Placeholder answer if execution error occurs.')
    parser.add_argument('--vote_method', type=str, default='simple',
                        choices=['simple', 'prob', 'answer_biased'])
    parser.add_argument('--answer_biased', type=int, default=None,
                        help='The answer to be biased w. answer_biased_weight in majority vote.')
    parser.add_argument('--answer_biased_weight', type=float, default=None,
                        help='The weight of the answer to be biased in majority vote.')
    parser.add_argument('--eval_level', type=int, default=4)
    # debugging options
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug_id', type=int, default=-1)
    # parser.add_argument('--debug_id', type=int, default=2312)

    args = parser.parse_args()

    print("Args info")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
