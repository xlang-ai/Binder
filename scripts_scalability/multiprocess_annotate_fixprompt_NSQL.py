"""
Multiprocess annotate NSQL-Question pairs.
"""

import time
import openai
import json
import argparse
import copy
import os
import traceback
from typing import List, Tuple, Dict
import multiprocessing

from generation.generator import Generator
from generation.post_filter import PostFilter
from retrieval.retriever import Retriever
from utils.utils import load_data_split, pprint_dict
from nsql.database import NeuralDB

PROMPT_MAX_LENGTH = 8001 - 512
template = 'default_template'


def worker_annotate(
        pid: int,
        args,
        generator: Generator,
        retriever: Retriever,
        g_eids: List,
        dataset,
        tokenizer
):
    """
    A worker process for annotating.
    """
    g_dict, few_shot_retrieve_cache = dict(), dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:
        try:
            g_data_item = dataset[g_eid]
            g_dict[g_eid] = {
                'generations': dict(),
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}],
                eid=g_eid
            )
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()
            few_shot_retrieve_cache[g_eid] = dict()
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=os.path.join(args.template_dir, 'prompts/', args.prompt_file),
                num_shots=args.num_shots
            )
            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                phase='generate',
                generate_type=('nsql',),
                retrieve_content=args.retrieve_content,
                keep_row_order=args.keep_row_order,
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt

            # Ensure the input length fit Codex max input tokens
            num_shots, num_rows_to_remain = args.num_shots, 200
            while len(tokenizer.tokenize(prompt)) >= PROMPT_MAX_LENGTH:
                print(len(tokenizer.tokenize(prompt)))
                if args.prompt_shrink_method == 'shrink_shots':
                    num_shots -= 1
                    assert num_shots >= 0
                    few_shot_prompt = generator.build_few_shot_prompt_from_file(
                        file_path=os.path.join(args.template_dir, 'prompts/', args.prompt_file),
                        num_shots=num_shots
                    )
                    prompt = few_shot_prompt + "\n\n" + generate_prompt
                elif args.prompt_shrink_method == 'shrink_rows':
                    prompt = generator.prompt_row_truncate(prompt, num_rows_to_remain, table_end_token='*/')
                    num_rows_to_remain -= 20

            print(f"Process#{pid}: Building prompt for eid#{g_eid}, wtqid#{g_data_item['id']}")
            built_few_shot_prompts.append((g_eid, prompt))
            if len(built_few_shot_prompts) < args.num_parallel_prompts:
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                phase='generate',
                generate_type=('nsql',),
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                g_dict[eid]['generations'][template] = g_pairs

            built_few_shot_prompts = []
        except Exception as e:
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e}")
            print(traceback.print_exc())

    # final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            phase='generate',
            generate_type=('nsql',),
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            g_dict[eid]['generations'][template] = g_pairs

    return g_dict, few_shot_retrieve_cache


def worker_filter(
        pid: int,
        args,
        generator: Generator,
        post_filter: PostFilter,
        g_dict: Dict,
        dataset,
        few_shot_retrieve_cache: Dict
):
    """
    A worker process for filtering.
    """
    built_few_shot_prompts, candi_g_pairs = [], dict()
    num_generations_filtered = 0
    for g_eid, g_dict_per_sample in g_dict.items():
        try:
            ori_data_item = g_dict_per_sample['ori_data_item']
            ori_question = ori_data_item['question']
            g_data_item = copy.deepcopy(ori_data_item)
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()
            g_pairs = g_dict_per_sample['generations'][template]
            g_pairs = post_filter.coarse_filter(
                g_data_item=g_data_item,
                g_pairs=g_pairs,
                num_sort_by_prob=args.num_sort_by_prob,
                allow_pure_sql=args.allow_pure_sql,
                verbose=args.verbose
            )
            g_pairs_filtered = []

            if args.use_back_translation:  # FIXME: Don't use it now!
                if args.use_retriever:
                    few_shot_nsqls = few_shot_retrieve_cache[g_eid][template]

                few_shot_prompt = post_filter.build_few_shot_prompt(
                    dataset=dataset,
                    template=template,
                    nsqls=few_shot_nsqls,
                    phase='filter',
                    prompt_type=('nsql', 'question'),  # nsql & question (in order) in prompt
                    retrieve_content=args.retrieve_content,
                    keep_row_order=args.keep_row_order,
                )

                for idx, (g_nsql, _, logprob) in enumerate(g_pairs):
                    g_data_item['nsql'] = g_nsql
                    generate_prompt = generator.build_generate_prompt(
                        data_item=g_data_item,
                        phase='filter',
                        generate_type=('question',),
                        retrieve_content=args.retrieve_content,
                        keep_row_order=args.keep_row_order,
                    )
                    prompt = few_shot_prompt + generate_prompt

                    sid = f'{g_eid}-{idx}'
                    print(f"Process#{pid}: Building filter prompt for eid#{sid}, wtqid#{g_data_item['id']}")
                    built_few_shot_prompts.append((sid, prompt))
                    candi_g_pairs[sid] = (g_nsql, ori_question)
                    if len(built_few_shot_prompts) < args.num_parallel_prompts and idx != len(g_pairs) - 1:
                        continue

                    print(
                        f"Process#{pid}: Prompts (filter) ready with {len(built_few_shot_prompts)} parallels. "
                        f"Run openai API.")
                    check_bt_results = post_filter.check_back_translation(
                        phase='filter',
                        generate_type=('question',),  # generate question
                        prompts=built_few_shot_prompts,
                        candi_g_pairs=candi_g_pairs,
                        verbose=args.verbose
                    )

                    for sid, valid in check_bt_results.items():
                        if valid:
                            g_pairs_filtered.append(candi_g_pairs[sid])

                    built_few_shot_prompts = []
                    candi_g_pairs = dict()
            else:
                g_pairs_filtered = g_pairs

            g_dict_per_sample['generations'][template] = g_pairs_filtered
            num_generations_filtered += len(g_pairs_filtered)
            print(f"Process#{pid}: Filtering retains {len(g_pairs_filtered)}/{len(g_pairs)} generations.")
        except Exception as e:
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} filtering error: {e}")

    return g_dict


def main():
    # load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    # load openai keys
    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

    # annotate
    generator = Generator(args, keys=keys)
    retriever = Retriever(args)
    generate_eids = list(range(len(dataset)))

    if args.debug_eid != -1:  # TODO: debug
        generate_eids = [int(eid) for eid in args.debug_eid.split(';;')]
        # wtq2eid_map = {dataset[idx]['id']: idx for idx in range(len(dataset))}
        # generate_eids = [wtq2eid_map[args.debug_eid]]

    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)

    print('\n******* Annotating *******')
    g_dict, few_shot_retrieve_cache = dict(), dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="../utils/gpt2")
        worker_results.append(pool.apply_async(worker_annotate, args=(
            pid,
            args,
            generator,
            retriever,
            generate_eids_group[pid],
            dataset,
            tokenizer
        )))

    # merge annotation results
    for r in worker_results:
        worker_g_dict, worker_few_shot_retrieve_cache = r.get()
        g_dict.update(worker_g_dict)
        few_shot_retrieve_cache.update(worker_few_shot_retrieve_cache)
    pool.close()
    pool.join()

    # save unfiltered
    with open(os.path.join(args.save_dir, f'unfiltered_{args.dataset}_nsqls.json'), 'w') as f:
        json.dump(g_dict, f, indent=4)

    # post filter
    post_filter = PostFilter(args, keys=keys)
    g_dict_group = [dict() for _ in range(args.n_processes)]
    for g_eid in g_dict:
        g_dict_group[int(g_eid) % args.n_processes][g_eid] = g_dict[g_eid]

    print('\n********* Filtering ***********')
    if args.use_filter:
        filtered_g_dict = dict()
        worker_results = []
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_filter, args=(
                pid,
                args,
                generator,
                post_filter,
                g_dict_group[pid],
                dataset,
                few_shot_retrieve_cache
            )))

        # merge filter results
        for r in worker_results:
            worker_filtered_g_dict = r.get()
            filtered_g_dict.update(worker_filtered_g_dict)
        pool.close()
        pool.join()

        # save filtered
        with open(os.path.join(args.save_dir, f'filtered_{args.dataset}_nsqls.json'), 'w') as f:
            json.dump(filtered_g_dict, f, indent=4)

    # log
    num_generations = len(generate_eids) * args.num_generations_per_sample * args.sampling_n
    print(f"\nAnnotate {num_generations} nsql-question pairs for {len(generate_eids)} tables.\n"
          f"Elapsed time: {time.time() - start_time}")


if __name__ == '__main__':
    import platform, multiprocessing

    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # file path or name
    parser.add_argument('--dataset', type=str, default='wikitq_scalability_500rows',
                        choices=['wikitq_scalability_ori',
                                 'wikitq_scalability_100rows',
                                 'wikitq_scalability_200rows',
                                 'wikitq_scalability_500rows'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--template_dir', type=str, default='../templates/')
    parser.add_argument('--save_dir', type=str, default='../results_scalability/')
    parser.add_argument('--prompt_file', type=str, default='prompt_w_sql_v3_no_CoT.txt')
    parser.add_argument('--api_keys_file', type=str, default='../key.txt')

    # multiprocess options
    parser.add_argument('--n_processes', type=int, default=4)

    # nsql generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table_select_3_hidden',
                                 'create_table'])
    parser.add_argument('--num_shots', type=int, default=15)
    parser.add_argument('--num_generations_per_sample', type=int, default=1)
    parser.add_argument('--prompt_shrink_method', type=str, default='shrink_rows',
                        choices=['shrink_shots', 'shrink_rows'])
    parser.add_argument('--retrieve_content', action='store_true')
    parser.add_argument('--keep_row_order', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # nsql generation control options
    parser.add_argument('--ctr_target_columns', action='store_true')
    parser.add_argument('--ctr_target_columns_strategy', type=str, default='random',
                        choices=['random', 'traverse'])
    parser.add_argument('--ctr_operators', action='store_true')
    parser.add_argument('--ctr_operators_strategy', type=str, default='random',
                        choices=['random', 'traverse'])
    parser.add_argument('--ctr_nested_levels', action='store_true')
    parser.add_argument('--ctr_nested_levels_strategy', type=str, default='fixed',
                        choices=['fixed', 'random', 'traverse'])

    # nsql retrieve options
    parser.add_argument('--use_retriever', action='store_true')
    parser.add_argument('--retrieve_method', type=str, default='qh2qh_bleu',
                        choices=['q2q_bleu', 'q2q_ngram', 'qh2qh_bleu'])

    # nsql filter options
    parser.add_argument('--use_filter', action='store_false')
    parser.add_argument('--use_back_translation', action='store_true')
    parser.add_argument('--num_sort_by_prob', type=int, default=20)
    parser.add_argument('--max_same_keywords', type=int, default=10,
                        help='Max #generations with the same nsql keywords.')
    parser.add_argument('--allow_pure_sql', action='store_false')

    # codex options
    parser.add_argument('--engine', type=str, default="code-davinci-002")
    parser.add_argument('--num_parallel_prompts', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=20)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    # debug options
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug_eid', type=str, default=-1)
    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')

    main()
