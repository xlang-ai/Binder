import json
import numpy as np
import os
import argparse
from transformers import AutoTokenizer
import stanza
import nltk
import random
import pandas as pd
from typing import List, Dict
import re
from nltk.metrics.distance  import edit_distance
nltk.download('words')
from nltk.corpus import words

from generation.generator import Generator
from utils.utils import load_data_split, pprint_dict
from utils.matcher import Matcher
from nsql.database import NeuralDB

PROMPT_MAX_LENGTH = 8001 - 512
random.seed(42)


class Disturber(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def disturb(
            self,
            table: pd.DataFrame,
            method='default',
            phrase2matched_cells=None
    ):
        if method == 'default':
            return self._disturb_with_distractor(table, phrase2matched_cells)
        else:
            raise ValueError

    def _disturb_with_distractor(
            self,
            table: pd.DataFrame,
            phrase2matched_cells: Dict[str, List]
    ):
        matched_cell2column_id = {v[0][0]: v[0][-1][1] for k, v in phrase2matched_cells.items()}
        n_rows = len(table)
        for matched_cell, column_id in matched_cell2column_id.items():
            dtype = table.dtypes[column_id]
            new_cell = matched_cell
            if dtype == 'int64' or dtype == 'float64':
                number_disturb_range = [-2, -1, 1, 2]
                new_cell = float(matched_cell) + random.choice(number_disturb_range)
            elif dtype == 'datetime64[ns]':
                # TODO
                pass
            else:
                correct_words = words.words()
                temp = [(edit_distance(matched_cell, w), w) for w in correct_words if w[0] == matched_cell[0]]
                for candidate in sorted(temp, key=lambda val: val[0]):
                    if candidate[1] != matched_cell:
                        new_cell = candidate[1]
                        break
            for row_id in range(n_rows):
                NUMBER_PATTERN = re.compile('^[+-]?([0-9]*[.])?[0-9]+$')
                NUMBER_UNITS_PATTERN = re.compile('^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
                DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
                cell = table.iloc[row_id, column_id]
                if str(cell) == matched_cell:
                    continue
                # 15% replace a cell with "distractor" content relevant to the question
                if random.random() < 0.15:
                    table.iloc[row_id, column_id] = new_cell
                    print(f"Perform {table.dtypes[column_id]} type disturb at position ({row_id},{column_id}) : {cell} -> {new_cell}")
        return table


def main():
    # load openai keys
    with open(args.api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    dataset = load_data_split('wikitq', 'validation')
    generator = Generator(args, keys)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="../utils/gpt2")
    disturber = Disturber(args, tokenizer)
    matcher = Matcher()
    debug_ids = ['nt-355']
    n_disturb_tables = 0
    for eid, data_item in enumerate(dataset):
        if data_item['id'] not in debug_ids:
            continue
        try:
            print('-' * 80)
            print('id: ', eid, data_item['id'])
            print('question: ', data_item['question'])
            print('gold answer: ', data_item['answer_text'])
            db = NeuralDB(
                tables=[{'title': data_item['table']['page_title'], 'table': data_item['table']}],
            )
            data_item['table'] = db.get_table_df()
            data_item['title'] = db.get_table_title()
            phrase2matched_cells = matcher.match_sentence_with_table(data_item['question'], data_item['table'])
            print("phrase2matched_cells:", phrase2matched_cells)
            if len(phrase2matched_cells) == 0:
                continue
            n_disturb_tables += 1
            data_item['table'] = disturber.disturb(
                table=data_item['table'],
                phrase2matched_cells=phrase2matched_cells
            )
            print(f'n_disturb_tables: {n_disturb_tables}/{len(dataset)}')
            # # Save
            # _, table_subdir, table_name = data_item['table_id'].split('/')
            # table_name = table_name[:-4]
            # save_path = f"{args.table_dir}/{table_subdir}/{table_name}_{data_item['id']}.tsv"
            # data_item['table'].to_csv(save_path, sep="\t", index=False)
        except Exception as e:
            print(f"Disturb error: {e}")
    print(f'n_disturb_tables: {n_disturb_tables}/{len(dataset)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['has_squall',
                                 'missing_squall',
                                 'wikitq',
                                 'wikitq_sql_solvable',
                                 'wikitq_sql_unsolvable',
                                 'wikitq_sql_unsolvable_but_in_squall',
                                 'tab_fact',
                                 'hybridqa'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--template_dir', type=str, default='../templates/')
    parser.add_argument('--save_dir', type=str, default='../results_wikitq_full/')
    # parser.add_argument('--prompt_file', type=str, default='prompt_w_sql_v3_no_CoT.txt')
    parser.add_argument('--prompt_file', type=str, default='prompt_qa_balanced.txt')
    parser.add_argument('--api_keys_file', type=str, default='../key.txt')
    parser.add_argument('--table_dir', type=str,
                        default='../datasets/data/downloads/extracted/711cd80f2d751dffbf163586e177e0f1c063b3d7220aa286cd7630c8d70cd06c/WikiTableQuestions-master/disturbed_csv')

    # multiprocess options
    parser.add_argument('--n_processes', type=int, default=4)

    # nsql generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table_select_3_hidden',
                                 'create_table'])
    parser.add_argument('--num_shots', type=int, default=15)
    parser.add_argument('--num_generations_per_sample', type=int, default=1)
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
