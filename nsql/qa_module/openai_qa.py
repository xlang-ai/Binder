import backoff
import openai
import copy
from typing import List
import sys
import os
import random

from generation.prompt import OpenAIQAPromptBuilder
from generation.generator import Generator
from retrieval.retriever import OpenAIQARetriever
from retrieval.retrieve_pool import OpenAIQARetrievePool, QAItem

num_parallel_prompts = 10
num_qa_shots = 8
infinite_rows_len = 50  # If the table contain rows larger than this number, it will be handled rows by rows.
max_tokens = 1024


class OpenAIQAModel(object):
    def __init__(self):
        super().__init__()
        self.model = "code"
        assert self.model in ['text', 'code']
        self.type_options = ['Number', 'Boolean', 'DateTime', 'Others']  # Find it useless, aborted

        self.key_current_id = 0
        with open("keys.txt", "r") as f:
            self.keys = [line.strip() for line in f.readlines()]
        random.seed(42)
        random.shuffle(self.keys)

        path_of_this_file = os.path.dirname(__file__)
        root_dir = os.path.join(path_of_this_file, "../../")
        retrieve_pool = OpenAIQARetrievePool(
            data_path=os.path.join(root_dir, 'templates/', 'qa_retrieve_pool.json') # change to mmqa_qa_retrieve_pool_v2
        )
        self.retriever = OpenAIQARetriever(retrieve_pool)
        self.generator = Generator(args=None, keys=self.keys)  # Just to use its call api function

        self.prompting_method = 'new_db'
        self.answer_split_token: str = ';'
        self.db_mapping_token = "\t"


    def call_openai_api_completion(self, prompt):
        completion = self.generator._call_codex_api(engine="{}-davinci-002".format(self.model),
                                                    prompt=prompt,
                                                    max_tokens=max_tokens,
                                                    temperature=0,
                                                    top_p=1,
                                                    n=1,
                                                    stop=["\n\n"])
        return completion

    def call_openai_for_completion_text(self, prompt, openai_usage_type="completion"):
        if openai_usage_type == "completion":
            completion = self.call_openai_api_completion(prompt)
            return completion.choices[0].text
        elif openai_usage_type == "insert":
            completion = self.call_openai_api_insert(prompt)
            return completion.choices[0].text
        elif openai_usage_type == "edit":
            completion = self.call_openai_api_edit(prompt)
            return completion.choices[0].text[len(prompt):]
        else:
            raise ValueError("The model usage type '{}' doesn't exists!".format(openai_usage_type))

    @staticmethod
    def merge_tables(tables, by='row_id'):
        assert len(set([len(_table['rows']) for _table in tables])) == 1, "Tables must have the same rows!"
        merged_header = [by]
        try:
            by_idx = tables[0]['header'].index(by)
        except:
            print("tingting")
        merged_rows = [[_row[by_idx]] for _row in tables[0]['rows']]

        for _table in tables:
            header, rows = _table['header'], _table['rows']
            for col_idx, col in enumerate(header):
                if col == by:
                    continue
                if col in merged_header:
                    # When the column is duplicate, and postfix _0, _1 etc.
                    col = "{}_{}".format(col, merged_header.count(col))
                merged_header.append(col)
                for i, row in enumerate(rows):
                    merged_rows[i].append(row[col_idx])
        return {"header": merged_header, "rows": merged_rows}

    def get_return_type(self, question: str, other_context: str, type_options):
        # Find it useless, aborted
        # For now, we use LLM for the classification of question type.
        # todo: the question starts with 'which' is always be recognized to 'boolean', we should find a way to fix that.
        prompt = ''
        prompt += 'Given {}\n'.format(other_context)
        prompt += "And question: '{}'\n\n".format(question)
        prompt += 'Q: Which kind of answer type we should return? '
        prompt += 'Choose from {}.\n'.format(type_options)
        prompt += 'A:'

        completion_text = self.call_openai_for_completion_text(prompt)
        print("Return answer type is recognized as: ", completion_text.lower().strip())
        return completion_text.lower().strip()

    def paraphrase_question(self, question: str, other_context: str, ):
        # Find it useless, aborted
        # We use LLM to paraphrase better question for itself to answer.
        prompt = ''
        prompt += 'Given {}\n'.format(other_context)
        prompt += "Q: What do we mean by \"{}\" in the given table, give us a more proper question.".format(question)
        prompt += 'A:'

        completion_text = self.call_openai_for_completion_text(prompt)
        print("LLM paraphrase the question into '", completion_text.lower().strip(), "'")
        return completion_text

    def wrap_with_prompt_for_simple_qa(self, question, header, row, sub_table, auxiliary_prompt=True,
                                       previous_return_type=None, eid=-1):
        prompt = ""
        context = ", ".join(["{} {}".format(h, r) for h, r in zip(header, row)])
        prompt += "Q: Given {}, answer question \"{}\"".format(context, question)
        if auxiliary_prompt:
            return_type = self.get_return_type(question, "",
                                               self.type_options) if not previous_return_type else previous_return_type
            return_type = return_type.lower()
            if return_type == "number":
                prompt += " Return a Number.\n"
            elif return_type == "boolean":
                prompt += " Choose from 'Yes' or 'No'.\n"
            elif return_type == "datetime":
                prompt += " Return a Datetime formatted as 'yyyy-mm-dd hh:mm:ss'.\n"
            else:  # No specific restriction on return type.
                prompt += "\n"
        else:
            prompt += "The answer should be a word.\n"
        prompt += "A:"

        return prompt, return_type

    def wrap_with_prompt_for_table_qa(self,
                                      question,
                                      sub_table,
                                      table_title=None,
                                      answer_split_token=None,
                                      qa_type="ans",
                                      prompting_method="new_db",
                                      db_mapping_token="😅",
                                      auxiliary_prompt=False,
                                      eid=-1,
                                      verbose=True):
        # QA few-shot added by @zhoujun
        prompt = "Question Answering Over Database:\n\n"
        if qa_type in ['map', 'ans'] and num_qa_shots > 0:
            query_item = QAItem(id=eid, qa_question=question, table=sub_table, title=table_title)
            retrieved_items = self.retriever.retrieve(item=query_item, num_shots=num_qa_shots, qa_type=qa_type)
            few_shot_prompt_list = []
            for item in retrieved_items:
                one_shot_prompt = OpenAIQAPromptBuilder.build_one_shot_prompt(
                    item=item,
                    answer_split_token=answer_split_token,
                    verbose=verbose,
                    prompting_method=prompting_method,
                    db_mapping_token=db_mapping_token
                )
                few_shot_prompt_list.append(one_shot_prompt)
            few_shot_prompt = '\n'.join(few_shot_prompt_list[:num_qa_shots])
            prompt = few_shot_prompt

        prompt += "\nGive a database as shown below:\n{}\n\n".format(
            OpenAIQAPromptBuilder.table2codex_prompt(sub_table, table_title)
        )

        if qa_type == "map":
            prompt += "Q: Answer question \"{}\" row by row.".format(question)
            assert answer_split_token is not None
            if prompting_method == "basic":
                prompt += " The answer should be a list split by '{}' and have {} items in total.".format(
                    answer_split_token, len(sub_table['rows']))

        elif qa_type == "ans":
            prompt += "Q: Answer question \"{}\" for the table.".format(question)
            prompt += " "
        else:
            raise ValueError("The QA type is not supported!")

        if auxiliary_prompt:
            return_type = self.get_return_type(question,
                                               OpenAIQAPromptBuilder.table2codex_prompt(sub_table, table_title),
                                               self.type_options)
            return_type = return_type.lower()
            if qa_type == "map":
                if return_type == "number":
                    prompt += " Each item is a Number.\n"
                elif return_type == "boolean":
                    prompt += " Each item is chosen from 'Yes' or 'No'.\n"
                elif return_type == "datetime":
                    prompt += " Each item is a Datetime formatted as 'yyyy-mm-dd hh:mm:ss'.\n"
                else:  # No specific restriction on return type.
                    prompt += "\n"

            elif qa_type == "ans":
                if return_type == "number":
                    prompt += " Return a Number.\n"
                elif return_type == "boolean":
                    prompt += " Choose from 'Yes' or 'No'.\n"
                elif return_type == "datetime":
                    prompt += " Return a Datetime formatted as 'yyyy-mm-dd hh:mm:ss'.\n"
                else:  # No specific restriction on return type.
                    prompt += "\n"
            else:
                raise ValueError("The QA type is not supported!")
        else:
            prompt += "\n"
        if qa_type == "map":
            if prompting_method == "basic":
                prompt += "A:"
        elif qa_type == "ans":
            prompt += "A:"

        return prompt

    def separate_judge(self, question, merged_table, qa_type):
        # Find it useless, aborted
        return len(merged_table['rows']) > 100

    def qa(self, question, sub_tables, qa_type: str, eid: int, verbose: bool = True, **args):
        # If it is not a problem API can handle, answer it with a QA model.
        merged_table = OpenAIQAModel.merge_tables(sub_tables)
        separate = self.separate_judge(question, merged_table, qa_type)
        separate = False
        if verbose:
            print("Make Question {} on {}".format(question, merged_table))
        if qa_type == "map":
            # Map: col(s) -question> one col
            if separate:
                # Make model make QAs towards each row(parameter pairs)
                # col(s) -> one col, inner QA
                _rows = merged_table['rows']
                _header = merged_table['header']
                answers = []
                prompt_list = []
                return_type = None
                for _row in _rows:
                    # Drop the row_id for simple qa, keep the whole table as context.
                    prompt, return_type = self.wrap_with_prompt_for_simple_qa(question, _header[1:], _row[1:],
                                                                              merged_table,
                                                                              previous_return_type=return_type,
                                                                              eid=eid)
                    prompt_list.append(prompt)
                answers = [completion.text.strip().lower() for completion in
                           self.call_openai_for_list_of_text_completion(prompt_list)]
                # print("COMPLETION", completion_str)
                # answers.append(completion_str)

            else:
                # Make model make a QA towards a sub-table
                # col(s) -> one col, all QA in one time

                def do_map(_table):
                    _prompt = self.wrap_with_prompt_for_table_qa(question,
                                                                 _table,
                                                                 args['table_title'],
                                                                 self.answer_split_token,
                                                                 qa_type,
                                                                 eid=eid,
                                                                 prompting_method=self.prompting_method,
                                                                 db_mapping_token=self.db_mapping_token,
                                                                 verbose=verbose)
                    completion_str = self.call_openai_for_completion_text(_prompt).lower().strip(' []')

                    if verbose:
                        print(f'QA map@ input:\n{_prompt}')
                        print(f'QA map@ output:\n{completion_str}')

                    if self.prompting_method == "basic":
                        answers = [_answer.strip(" '").lower() for _answer in
                                   completion_str.split(self.answer_split_token)]
                    elif self.prompting_method == "new_db":
                        answers = [line.split(self.db_mapping_token)[-1] for line in completion_str.split("\n")[2:-1]]
                    else:
                        raise ValueError("No such prompting methods: '{}'! ".format(self.prompting_method))
                    return answers

                # Handle infinite rows, rows by rows.
                answers = []
                rows_len = len(merged_table['rows'])
                run_times = int(rows_len / infinite_rows_len) if rows_len % infinite_rows_len == 0 else int(
                    rows_len / infinite_rows_len) + 1

                for run_idx in range(run_times):
                    _table = {
                        "header": merged_table['header'],
                        "rows": merged_table['rows'][run_idx * infinite_rows_len:]
                    } if run_idx == run_times - 1 else \
                        {
                            "header": merged_table['header'],
                            "rows": merged_table['rows'][run_idx * infinite_rows_len:(run_idx + 1) * infinite_rows_len]
                        }

                    answers.extend(do_map(_table))
            if verbose:
                print("The map@ openai answers are {}".format(answers))
            # Add row_id in addition for finding to corresponding rows.
            return {"header": ['row_id'] + args['new_col_name_s'],
                    "rows": [[row[0], answer] for row, answer in zip(merged_table['rows'], answers)]}
        elif qa_type == "ans":
            # Ans: col(s) -question> answer
            prompt = self.wrap_with_prompt_for_table_qa(question,
                                                        merged_table,
                                                        args['table_title'],
                                                        eid=eid,
                                                        prompting_method=self.prompting_method,
                                                        verbose=verbose)
            answers = [self.call_openai_for_completion_text(prompt).lower().strip(' []')]

            if verbose:
                print(f'QA ans@ input:\n{prompt}')
                print(f'QA ans@ output:\n{answers}')

            return answers
        else:
            raise ValueError("Please choose from map and ans in the qa usage!!")
