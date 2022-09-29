"""
General utilities.
"""
import re
import json
import os
from typing import List, Union, Dict
from functools import cmp_to_key
import math
from collections.abc import Iterable

from datasets import load_dataset
from utils.normalizer import str_normalize, maybe_normalize_float
from utils.wikitq_evaluator import to_value_list, check_denotation


def majority_vote(
        nsqls: List,
        pred_answer_list: List,
        allow_none_and_empty_answer: bool = False,
        allow_error_answer: bool = False,
        answer_placeholder: Union[str, int] = '<error|empty>',
        vote_method: str = 'prob',
        answer_biased: Union[str, int] = None,
        answer_biased_weight: float = None,
        lf_biased_weight_dict: Dict[str, float] = None
):
    """
    Determine the final nsql execution answer by majority vote.
    """

    def _compare_answer_vote_simple(a, b):
        """
        First compare occur times. If equal, then compare max nsql logprob.
        """
        if a[1]['count'] > b[1]['count']:
            return 1
        elif a[1]['count'] < b[1]['count']:
            return -1
        else:
            if a[1]['nsqls'][0][1] > b[1]['nsqls'][0][1]:
                return 1
            elif a[1]['nsqls'][0][1] == b[1]['nsqls'][0][1]:
                return 0
            else:
                return -1

    def _compare_answer_vote_with_prob(a, b):
        """
        Compare prob sum.
        """
        return 1 if sum([math.exp(nsql[1]) for nsql in a[1]['nsqls']]) > sum(
            [math.exp(nsql[1]) for nsql in b[1]['nsqls']]) else -1

    # Vote answers
    candi_answer_dict = dict()
    for (nsql, _, logprob), pred_answer in zip(nsqls, pred_answer_list):
        if allow_none_and_empty_answer:
            if pred_answer == [None] or pred_answer == []:
                pred_answer = [answer_placeholder]
        if allow_error_answer:
            if pred_answer == '<error>':
                pred_answer = [answer_placeholder]

        # Invalid execution results
        if pred_answer == '<error>' or pred_answer == [None] or pred_answer == []:
            continue
        if candi_answer_dict.get(tuple(pred_answer), None) is None:
            candi_answer_dict[tuple(pred_answer)] = {
                'count': 0,
                'nsqls': []
            }
        answer_info = candi_answer_dict.get(tuple(pred_answer), None)
        answer_info['count'] += 1
        answer_info['nsqls'].append([nsql, logprob])

    # All candidates execution errors
    if len(candi_answer_dict) == 0:
        return answer_placeholder, [(nsqls[0][0], nsqls[0][-1])]

    # Sort
    if vote_method == 'simple':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'prob':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_prob), reverse=True)
    elif vote_method == 'answer_biased':
        # Specifically for Tabfact entailed answer, i.e., `1`.
        # If there exists nsql that produces `1`, we consider it more significant because `0` is very common.
        assert answer_biased_weight is not None and answer_biased_weight > 0
        for answer, answer_dict in candi_answer_dict.items():
            if answer == (answer_biased,):
                answer_dict['count'] *= answer_biased_weight
                # for nsql_w_logprob in answer_dict['nsqls']:
                #     nsql_w_logprob[1] += math.exp(answer_biased_weight)
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'lf_biased':
        # Assign weights to different types of logic forms (lf) to control interpretability and coverage
        assert lf_biased_weight_dict is not None
        for answer, answer_dict in candi_answer_dict.items():
            count = 0
            for nsql, _ in answer_dict['nsqls']:
                if 'map@' in nsql:
                    count += lf_biased_weight_dict['nsql_map@']
                elif 'ans@' in nsql:
                    count += lf_biased_weight_dict['nsql_ans@']
                else:
                    count += lf_biased_weight_dict['sql']
            answer_dict['count'] = count
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    else:
        raise ValueError(f"Vote method {vote_method} is not supported.")

    pred_answer_info = sorted_candi_answer_list[0]
    pred_answer, pred_answer_nsqls = list(pred_answer_info[0]), pred_answer_info[1]['nsqls']
    return pred_answer, pred_answer_nsqls


def eval_tabfact_match(pred, gold):
    if isinstance(pred, list):
        pred = pred[0]
    pred, gold = str(pred), str(gold)
    return pred == gold


def eval_ex_match(pred, gold, level=3, question=None):
    pred = [str(p).lower().strip() for p in pred]
    gold = [str(g).lower().strip() for g in gold]

    if level == 1:
        # UnifiedSKG/Tapex eval
        pred = [maybe_normalize_float(p) for p in pred]
        gold = [maybe_normalize_float(g) for g in gold]
        return sorted(pred) == sorted(gold)
    elif level == 2:
        # WikiTQ eval
        pred = to_value_list(pred)
        gold = to_value_list(gold)
        return check_denotation(pred, gold)
    elif level == 3:
        # WikiTQ eval w. our string normalization using recognizer
        pred = [str_normalize(span) for span in pred]
        gold = [str_normalize(span) for span in gold]
        pred = to_value_list(pred)
        gold = to_value_list(gold)
        return check_denotation(pred, gold)
    elif level == 4:
        # Besides the evaluation of level#3, add two pre-match checks, especially for SQL:
        assert isinstance(question, str)
        question = re.sub('\s+', ' ', question).strip().lower()
        pred = [str_normalize(span) for span in pred]
        gold = [str_normalize(span) for span in gold]
        pred = sorted(list(set(pred)))
        gold = sorted(list(set(gold)))
        # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
        if len(pred) == 1 and len(gold) == 1:
            if (pred[0] == '0' and gold[0] == 'no') \
                    or (pred[0] == '1' and gold[0] == 'yes'):
                return True
            question_tokens = question.split()
            try:
                pos_or = question_tokens.index('or')
                token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                if (pred[0] == '0' and gold[0] == token_after_or) \
                        or (pred[0] == '1' and gold[0] == token_before_or):
                    return True
            except Exception as e:
                pass
        # (2) Number value (allow units) and Date substring match
        if len(pred) == 1 and len(gold) == 1:
            NUMBER_PATTERN = re.compile('^[+-]?([0-9]*[.])?[0-9]+$')
            NUMBER_UNITS_PATTERN = re.compile('^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
            DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
            DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
            p, g = pred[0], gold[0]
            # Restore `duration` type, e.g., from 'P3Y' -> '3'
            if re.match(DURATION_PATTERN, p):
                p = re.match(DURATION_PATTERN, p).group(2)
            if re.match(DURATION_PATTERN, g):
                g = re.match(DURATION_PATTERN, g).group(2)
            match = False
            num_flag, date_flag = False, False
            # # Turn negative number to positive  # FIXME: need this?
            # if re.match(NUMBER_PATTERN, p) and re.match(NUMBER_PATTERN, g):
            #     num_flag = True
            #     p = p.lstrip('-')
            # Number w. unit match after string normalization.
            # Either pred or gold being number w. units suffices it.
            if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
                num_flag = True
            # Date match after string normalization.
            # Either pred or gold being date suffices it.
            if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                date_flag = True
            if num_flag:
                p_set, g_set = set(p.split()), set(g.split())
                if p_set.issubset(g_set) or g_set.issubset(p_set):
                    match = True
            if date_flag:
                p_set, g_set = set(p.replace('-', ' ').split()), set(g.replace('-', ' ').split())
                if p_set.issubset(g_set) or g_set.issubset(p_set):
                    match = True
            if match:
                return True
        pred = to_value_list(pred)
        gold = to_value_list(gold)
        return check_denotation(pred, gold)
    elif level == 5:
        # Loosen the standard of level#4, especially by allowing substring match. Require human checks.
        assert isinstance(question, str)
        question = re.sub('\s+', ' ', question).strip().lower()
        pred = [str_normalize(span) for span in pred]
        gold = [str_normalize(span) for span in gold]
        pred = sorted(list(set(pred)))
        gold = sorted(list(set(gold)))
        # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
        if len(pred) == 1 and len(gold) == 1:
            if (pred[0] == '0' and (gold[0] == 'no' or gold[0] == 'false')) \
                    or (pred[0] == '1' and (gold[0] == 'yes' or gold[0] == 'true')):
                return True
            question_tokens = question.split()
            try:
                pos_or = question_tokens.index('or')
                token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                if (pred[0] == '0' and gold[0] == token_after_or) \
                        or (pred[0] == '1' and gold[0] == token_before_or):
                    return True
            except Exception as e:
                pass
        # (2) gold tokens is subset of pred tokens, and vice versa. e.g., shanghai (china) -> shanghai
        if len(pred) == 1 and len(gold) == 1:
            DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
            p, g = list(pred)[0], list(gold)[0]
            # Restore `duration` type
            if re.match(DURATION_PATTERN, p):
                p = re.match(DURATION_PATTERN, p).group(2)
            if re.match(DURATION_PATTERN, g):
                g = re.match(DURATION_PATTERN, g).group(2)
            match = False
            split_tokens = [' ', '$', '-', '%']  # metrics for text, currency, date, percentage
            # Subset test
            for split_token in split_tokens:
                p_set = set(p.split(split_token))
                g_set = set(g.split(split_token))
                if p_set.issubset(g_set) or g_set.issubset(p_set):
                    match = True
                    break
            if match:
                return True
        pred = to_value_list(pred)
        gold = to_value_list(gold)
        return check_denotation(pred, gold)
    else:
        raise ValueError(f"WikiTQ evaluation level{level} is not supported.")


def load_data_split(dataset_to_load, split, data_dir='../datasets/'):
    dataset_split_loaded = load_dataset(
        path=os.path.join(data_dir, "{}.py".format(dataset_to_load)),
        cache_dir=os.path.join(data_dir, "data"))[split]

    # unify names of keys
    if dataset_to_load in ['wikitq', 'has_squall', 'missing_squall',
                           'wikitq', 'wikitq_sql_solvable', 'wikitq_sql_unsolvable',
                           'wikitq_sql_unsolvable_but_in_squall',
                           'wikitq_scalability_ori',
                           'wikitq_scalability_100rows',
                           'wikitq_scalability_200rows',
                           'wikitq_scalability_500rows',
                           'wikitq_robustness'
                           ]:
        pass
    elif dataset_to_load == 'tab_fact':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['question'] = data_item['statement']
            data_item['answer_text'] = data_item['label']
            data_item['table']['page_title'] = data_item['table']['caption']
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == 'hybridqa':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['context'].split(' | ')[0]
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == 'mmqa':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['table']['title']
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    else:
        raise ValueError(f'{dataset_to_load} dataset is not supported now.')
    return dataset_split_loaded


def pprint_dict(dic):
    print(json.dumps(dic, indent=2))


def flatten(nested_list):
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def get_nsql_components(nsql: str):
    """
    Extract(loosely) some key components from nsql.
    """
    # TODO: can't parse nested QA now
    # regex match
    select_pattern = "SELECT\s+(COUNT|MAX|MIN|SUM|AVG)?\(?(.*?)\)?\s+"
    qa_pattern = "QA\((.*?)\)"
    qa_pattern_half = "QA\((.*?);"

    selects = re.findall(select_pattern, nsql)
    qas = re.findall(qa_pattern, nsql)
    qa_halfs = re.findall(qa_pattern_half, nsql)

    # extract select ops and columns
    select_operators, select_columns = [], []
    for op, col in selects:
        if op:
            select_operators.append(op)
        if col:
            select_columns.append(col)

    # extract qa prompts and columns
    qa_prompts, qa_columns = [], []
    for prompt in qa_halfs:
        if prompt:
            qa_prompts.append(prompt)
    for qa in qas:
        qa_components = qa.split(';')
        col = qa_components[1].strip()
        if col and not col.startswith('SELECT'):
            qa_columns.append(col)

    return {
        'select_operators': sorted(list(set(select_operators))),
        'select_columns': sorted(list(set(select_columns))),
        'qa_prompts': sorted(list(set(qa_prompts))),
        'qa_columns': sorted(list(set(qa_columns)))
    }