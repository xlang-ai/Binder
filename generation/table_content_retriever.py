import random
import pandas as pd
from transformers import AutoTokenizer


class TableContentRetriever(object):
    """
    Retrieve relevant table content (3 rows) from the table in order to build prompt
    """
    def __init__(self, use_tokenizer=False, model_name="gpt2"):
        self.echo = True
        self.use_tokenizer = use_tokenizer
        if use_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def retrieve_rows(self, df, question, strategy, row_num, method=None, keep_row_order=False):
        if strategy == "question":
            rows = self._retrieve_rows_based_on_question(df, question, row_num, method, keep_row_order)
        else:
            raise ValueError("Unsupported strategy! Should be: question, target_column_content")
        return rows

    def _retrieve_rows_based_on_question(self, df, question, row_num, method, keep_row_order=False):
        id2ngram = dict()
        if self.use_tokenizer:
            question_toks = self.tokenizer.tokenize(question.strip("?!.,"))
        else:
            question_toks = question.strip("?!.,").split(" ")
        question_ngram_list = []

        # ngram 1 to 4, could be improved by other means
        for ngram_num in range(1, 5):
            question_ngram_list.extend(create_ngram_list(question_toks, ngram_num=ngram_num))

        for row_id, row in df.iterrows():
            if self.use_tokenizer:
                row_toks = self.tokenizer.tokenize(row.to_string())
            else:
                row_toks = [str(cell).lower() for cell in row]

            row_ngram_list = []
            for ngram_num in range(1, 5):
                row_ngram_list.extend(create_ngram_list(row_toks, ngram_num=ngram_num))

            ngram_align = len(set(question_ngram_list) & set(row_ngram_list))

            if ngram_align != 0:
                id2ngram[row_id] = ngram_align

        id2ngram = sorted(id2ngram.items(), key=lambda x: x[1], reverse=True)
        if len(id2ngram) == 0:
            # select top row_num rows from the table
            retrieve_df = df.iloc[:row_num]
        elif len(id2ngram) > row_num:
            row_idx = []
            for i in range(row_num):
                row_idx.append(id2ngram[i][0])
            if keep_row_order:
                row_idx = sorted(row_idx)
            retrieve_df = df.iloc[row_idx]
        else:
            row_idx = []
            for item in id2ngram:
                row_idx.append(item[0])
            missing_idx_num = row_num - len(row_idx)
            row_idx += random.sample(list(set(range(len(df))) - set(row_idx)), missing_idx_num)
            if keep_row_order:
                row_idx = sorted(row_idx)
            retrieve_df = df.iloc[row_idx]

        return retrieve_df

def create_ngram_list(input_list, ngram_num):
    ngram_list = []
    if len(input_list) <= ngram_num:
        ngram_list.extend(input_list)
    else:
        for tmp in zip(*[input_list[i:] for i in range(ngram_num)]):
            tmp = " ".join(tmp)
            ngram_list.append(tmp)
    return ngram_list


def create_ngram_set(input_list, ngram_num):
    if len(input_list) <= ngram_num:
        return {tuple(list(input_list))}
    else:
        return set(zip(*[input_list[i:] for i in range(ngram_num)]))