"""
Retriever to retrieve relevant examples from annotations.
"""

import copy
from typing import Dict, List, Tuple, Any
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
import os
import json

from utils.normalizer import normalize
from retrieval.retrieve_pool import OpenAIQARetrievePool, QAItem


class Retriever(object):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _n_gram(tks, n):
        """
        Return token-level n-gram of given tokens.
        """
        return [set(tks[i: i + n]) for i in range(len(tks) - n + 1)]

    @staticmethod
    def _string_n_gram_overlap_score(q1: str, q2: str, stop_words=None, stemmer=None):
        """
        n-gram overlap score between two queries.
        """
        q1, q2 = normalize(q1), normalize(q2)
        q1_tks = [tk for tk in nltk.word_tokenize(q1) if tk not in stop_words]
        q2_tks = [tk for tk in nltk.word_tokenize(q2) if tk not in stop_words]
        if stemmer is not None:
            q1_tks = [stemmer.stem(tk) for tk in q1_tks]
            q2_tks = [stemmer.stem(tk) for tk in q2_tks]

        max_n_gram_length = min(len(q1_tks), len(q2_tks))
        if max_n_gram_length == 0:
            return 0

        score = 0
        q1_tks_set = set(q1_tks)
        for i in range(1, max_n_gram_length + 1):
            q2_n_grams = Retriever._n_gram(q2_tks, i)
            for q2_n_gram in q2_n_grams:
                if q2_n_gram.issubset(q1_tks_set):
                    score = max(score, i / len(q1_tks_set))
                    break
        return score

    @staticmethod
    def _string_bleu(q1: str, q2: str, stop_words=None, stemmer=None):
        """
        BLEU score.
        """
        q1, q2 = normalize(q1), normalize(q2)
        reference = [[tk for tk in nltk.word_tokenize(q1)]]
        candidate = [tk for tk in nltk.word_tokenize(q2)]
        if stemmer is not None:
            reference = [[stemmer.stem(tk) for tk in reference[0]]]
            candidate = [stemmer.stem(tk) for tk in candidate]

        chencherry_smooth = SmoothingFunction()  # bleu smooth to avoid hard behaviour when no ngram overlaps
        bleu_score = sentence_bleu(
            reference,
            candidate,
            weights=(0.25, 0.3, 0.3, 0.15),
            smoothing_function=chencherry_smooth.method1
        )
        return bleu_score

    def _qh2qh_similarity(
            self,
            data_item: Dict,
            candi_data_item_list: List,
            num_retrieve_samples: int,
            score_func: str,
            weight_h: float = 0.1,
            verbose: bool = False
    ):
        """
        Retrieve top K nsqls based on query&header to query&header similarities.
        """
        q = data_item['question']
        candi_q_list = [(d[0], d[1]['question']) for d in candi_data_item_list]

        h = ' '.join(data_item['table']['header'])
        candi_h_list = [(d[0], ' '.join(d[1]['table']['header'])) for d in candi_data_item_list]

        # Remove self in the retrieve pool
        for idx in range(len(candi_q_list)):
            if candi_q_list[idx][1] == q and candi_h_list[idx][1] == h:
                self_idx = idx
                candi_q_list.pop(self_idx)
                candi_h_list.pop(self_idx)
                break

        stemmer = SnowballStemmer('english')
        if score_func == 'bleu':
            retrieve_q_list = [(d[0], Retriever._string_bleu(q, d[1], stemmer=stemmer))
                               for d in candi_q_list]
            retrieve_h_list = [(d[0], Retriever._string_bleu(h, d[1], stemmer=stemmer))
                               for d in candi_h_list]
            # print(sorted(retrieve_q_list, key=lambda x: x[1], reverse=True))
            # print(sorted(retrieve_h_list, key=lambda x: x[1], reverse=True))
            retrieve_list = [(retrieve_q_list[idx][0], retrieve_q_list[idx][1] + weight_h * retrieve_h_list[idx][1])
                             for idx in range(len(retrieve_q_list))]
        else:
            raise ValueError
        retrieve_list = sorted(retrieve_list, key=lambda x: x[1], reverse=True)
        retrieve_list = retrieve_list[:num_retrieve_samples]
        # print(retrieve_list[0])

        if verbose:
            print(retrieve_list)

        retrieve_eid_list = [d[0] for d in retrieve_list]
        return retrieve_eid_list

    def retrieve(
            self,
            data_item: Dict,
            dataset,
            nsqls: Dict,
            num_retrieve_samples: int,
            method: str,
            verbose: bool = False
    ):
        """
        Retrieve most relevant annotated samples to the data item.
        """
        # TODO: Set retrieve pool as a class variable (like OpenAIQARetriever)
        candi_data_item_list = [(eid, dataset[int(eid)]) for eid in nsqls.keys()]

        if method == 'q2q_bleu':
            r_eid_list = self._qh2qh_similarity(
                data_item,
                candi_data_item_list,
                num_retrieve_samples,
                'bleu',
                weight_h=0,
                verbose=verbose
            )
            r_nsqls = {eid: nsqls[eid] for eid in r_eid_list}
        elif method == 'qh2qh_bleu':
            r_eid_list = self._qh2qh_similarity(
                data_item,
                candi_data_item_list,
                num_retrieve_samples,
                'bleu',
                verbose=verbose
            )
            r_nsqls = {eid: nsqls[eid] for eid in r_eid_list}
        else:
            raise ValueError(f'Retrieve method {method} is not supported.')

        return r_nsqls


class OpenAIQARetriever(object):
    def __init__(self, retrieve_pool: OpenAIQARetrievePool):
        self.retrieve_pool = retrieve_pool

    def _qh2qh_similarity(
            self,
            item: QAItem,
            num_retrieve_samples: int,
            score_func: str,
            qa_type: str,
            weight_h: float = 0.2,
            verbose: bool = False
    ):
        """
        Retrieve top K nsqls based on query&header to query&header similarities.
        """
        q = item.qa_question
        header_wo_row_id = copy.copy(item.table['header'])
        header_wo_row_id.remove('row_id')
        h = ' '.join(header_wo_row_id)
        stemmer = SnowballStemmer('english')
        if score_func == 'bleu':
            retrieve_q_list = [(d, Retriever._string_bleu(q, d.qa_question.split('@')[1], stemmer=stemmer))
                               for d in self.retrieve_pool if d.qa_question.split('@')[0] == qa_type]
            retrieve_h_list = [(d, Retriever._string_bleu(h, ' '.join(d.table['header']), stemmer=stemmer))
                               for d in self.retrieve_pool if d.qa_question.split('@')[0] == qa_type]
            # print(sorted(retrieve_q_list, key=lambda x: x[1], reverse=True)[0])
            # print(sorted(retrieve_h_list, key=lambda x: x[1], reverse=True)[0])
            retrieve_list = [(retrieve_q_list[idx][0], retrieve_q_list[idx][1] + weight_h * retrieve_h_list[idx][1])
                             for idx in range(len(retrieve_q_list))]
        else:
            raise ValueError
        retrieve_list = sorted(retrieve_list, key=lambda x: x[1], reverse=True)
        # Remove self in retrieve pool
        retrieve_list = list(map(lambda x: x[0],
                                 filter(lambda x: x[0].id != item.id, retrieve_list)))[:num_retrieve_samples]
        # print(retrieve_list[0])

        if verbose:
            print(retrieve_list)

        return retrieve_list

    def retrieve(
            self,
            item: QAItem,
            num_shots: int,
            method: str = 'qh2qh_bleu',
            qa_type: str = 'map',
            verbose: bool = False
    ) -> List[QAItem]:
        """
        Retrieve a list of relevant QA samples.
        """
        if method == 'qh2qh_bleu':
            retrieved_items = self._qh2qh_similarity(
                item=item,
                num_retrieve_samples=num_shots,
                score_func='bleu',
                qa_type=qa_type,
                verbose=verbose
            )
            return retrieved_items
        else:
            raise ValueError(f'Retrieve method {method} is not supported.')


if __name__ == '__main__':
    dataset = load_dataset(
        path="../datasets/{}.py".format('missing_squall'),
        cache_dir="../datasets/data")['validation']
    with open(os.path.join('../generation/templates/', f'templates_missing_squall.json'), 'r') as f:
        template_dict = json.load(f)

    target_eid = 44
    target_data_item = dataset[target_eid]
    nsqls = list(template_dict.values())[0]
    retriever = Retriever(args=None)
    r_nsqls = retriever.retrieve(
        target_data_item,
        dataset,
        nsqls,
        3,
        'q2q_bleu'
    )
    print(json.dumps(r_nsqls, indent=2))
