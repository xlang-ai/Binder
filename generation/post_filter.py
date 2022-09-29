"""
Post-filter the generated NSQLs.
"""
from typing import Dict, List, Any, Tuple
import sqlite3
import re
import json

from generation.generator import Generator
from utils.utils import get_nsql_components
from nsql.nsql_exec import NeuralDB


class PostFilter(Generator):
    """
    A post filter to filter out invalid generations.
    """
    def __init__(self, args, keys=None):
        super().__init__(args, keys)

    def _flat_str(self, string):
        """
        Flatten a string by lowercasing and removing space.
        """
        return re.sub('\s+', '', string.strip().lower())

    def _nsql_match(self, nsql1, nsql2):
        """
        Match(fuzzy) of two nsqls w/o regard to QA prompt.
        """
        if self._flat_str(nsql1) == self._flat_str(nsql2):
            return True

        qa_prompts1 = get_nsql_components(nsql1)['qa_prompts']
        qa_prompts2 = get_nsql_components(nsql2)['qa_prompts']
        for prompt in qa_prompts1:
            nsql1 = re.sub(prompt, '', nsql1)
        for prompt in qa_prompts2:
            nsql2 = re.sub(prompt, '', nsql2)

        return self._flat_str(nsql1) == self._flat_str(nsql2)

    def _question_match(self, q1: str, q2: str):
        """
        Match of two questions.
        """
        return self._flat_str(q1) == self._flat_str(q2)

    def _check_back_translation(
            self,
            phase: str,
            generate_type: Tuple,
            prompts: List[Tuple],
            candi_g_pairs: Dict,
            verbose: bool = False
    ):
        """
        Check if back translated nsql equals to g_nsql.
        """
        response_dict = self.generate_one_pass(
            prompts=prompts,
            phase=phase,
            generate_type=generate_type,
            verbose=verbose
        )

        if verbose:
            if generate_type == ('question',):
                print("generation:", candi_g_pairs)
                print("bt_question:", response_dict)
                print()
            elif generate_type == ('nsql',):
                print("generation    :", candi_g_pairs)
                print("bt_nsql   :", response_dict)
                print()
            else:
                raise ValueError(f'Generate type {generate_type} is not supported.')

        check_bt_results = dict()
        if generate_type == ('question',):
            for sid, bt_pairs in response_dict.items():
                g_question = candi_g_pairs[sid][1]
                for _, bt_q in bt_pairs:
                    if self._question_match(bt_q, g_question):
                        check_bt_results[sid] = True
                        break
                if not check_bt_results.get(sid, False):
                    check_bt_results[sid] = False
        elif generate_type == ('nsql',):
            for sid, bt_pairs in response_dict.items():
                g_nsql = candi_g_pairs[sid][0]
                for bt_nsql, _ in bt_pairs:
                    try:
                        if self._nsql_match(bt_nsql, g_nsql):
                            check_bt_results[sid] = True
                            break
                    except:
                        pass
                if not check_bt_results.get(sid, False):
                    check_bt_results[sid] = False
        else:
            raise ValueError(f'Generate type {generate_type} is not supported.')

        return check_bt_results

    def _check_simple_grammar(
            self,
            g_data_item: Dict,
            g_nsql: str,
            verbose: bool = False
    ):
        """
        Check if nsql passes simple grammar check.
        """
        # TODO
        return True

    def _check_execution(
            self,
            g_data_item: Dict,
            g_nsql: str,
            verbose: bool = False
    ):
        """
        Check if nsql can be successfully executed on db.
        """
        # TODO
        return True

    def _check_contain_QA(self, g_nsql: str):
        """
        Check if nsql contains QA module.
        """
        return "QA(" in g_nsql

    def check_back_translation(
            self,
            phase: str,
            generate_type: Tuple,
            prompts: List[Tuple],
            candi_g_pairs: Dict,
            verbose: bool = False
    ):
        """
        Check whether the generated nsql and question are valid.
        """
        check_results = self._check_back_translation(
                prompts=prompts,
                candi_g_pairs=candi_g_pairs,
                phase=phase,
                generate_type=generate_type,
                verbose=verbose
            )

        return check_results

    def coarse_filter(
            self,
            g_data_item: Dict,
            g_pairs: List,
            num_sort_by_prob: int,
            allow_pure_sql: bool,
            verbose: bool
    ):
        """
        Coarse filtering.
        """
        # 1. keep the generations with highest logprobs
        if num_sort_by_prob > 0:
            g_pairs = g_pairs[:num_sort_by_prob]

        g_pairs_filtered = []
        occur_set = set()
        keywords_counter = dict()
        for g_nsql, g_question, logprob in g_pairs:

            # 2. remove duplicates (simple normalization)
            if (''.join(g_nsql.strip(';').lower().split()), g_question) in occur_set:
                continue
            occur_set.add((''.join(g_nsql.strip(';').lower().split()), g_question))

            # 3. remove nsql without QA module
            if not allow_pure_sql and not self._check_contain_QA(g_nsql):
                continue

            # 4. constrain #nsqls with the same keywords
            try:
                components = get_nsql_components(g_nsql)
                keywords = f"select_op:{';'.join(components['select_operators'])}||" \
                           f"select_col:{';'.join(components['select_columns'])}||" \
                           f"qa_col:{';'.join(components['qa_columns'])}"
                count = keywords_counter.get(keywords, 0)
                count += 1
                keywords_counter[keywords] = count
                if count > self.args.max_same_keywords:
                    continue
            except Exception as e:
                if verbose:
                    print(e)

                continue

            # 5. check simple grammar
            if not self._check_simple_grammar(
                g_data_item,
                g_nsql,
                verbose
            ):
                continue

            # 6. check if nsql can be executed successfully
            if not self._check_execution(
                g_data_item,
                g_nsql,
                verbose
            ):
                continue

            g_pairs_filtered.append((g_nsql, g_question, logprob))

        if verbose:
            print(json.dumps(keywords_counter, indent=2))

        return g_pairs_filtered



