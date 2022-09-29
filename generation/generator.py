"""
Generate nsql and questions.
"""

from typing import Dict, List, Union, Tuple
import openai
import time

from generation.prompt import PromptBuilder

class Generator(object):
    """
    Codex generation wrapper.
    """

    def __init__(self, args, keys=None):
        self.args = args
        self.keys = keys
        self.current_key_id = 0

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

    def prompt_row_truncate(
            self,
            prompt: str,
            num_rows_to_remain: int,
            table_end_token: str = '*/',
    ):
        """
        Fit prompt into max token limits by row truncation.
        """
        table_end_pos = prompt.rfind(table_end_token)
        assert table_end_pos != -1
        prompt_part1, prompt_part2 = prompt[:table_end_pos], prompt[table_end_pos:]
        prompt_part1_lines = prompt_part1.split('\n')[::-1]
        trunc_line_index = None
        for idx, line in enumerate(prompt_part1_lines):
            if '\t' not in line:
                continue
            row_id = int(line.split('\t')[0])
            if row_id <= num_rows_to_remain:
                trunc_line_index = idx
                break
        new_prompt_part1 = '\n'.join(prompt_part1_lines[trunc_line_index:][::-1])
        prompt = new_prompt_part1 + '\n' + prompt_part2
        return prompt

    def build_few_shot_prompt(
            self,
            dataset,
            template: str,
            nsqls: Dict,
            phase: str,
            prompt_type: Tuple,
            num_shots: int = None,
            retrieve_content: bool = False,
            keep_row_order: bool = False,
    ):
        """
        Build few-shot prompt for generation.
        """

        few_shot_prompt_list = []
        for eid, nsql_dict in nsqls.items():
            eid = int(eid)
            data_item = dataset[eid]
            data_item['eid'] = eid
            data_item['nsql'] = nsql_dict['nsql']
            data_item['target_columns'] = nsql_dict['target_columns']
            data_item['operators'] = nsql_dict['operators']
            data_item['nested_levels'] = nsql_dict['nested_levels']
            one_shot_prompt = self.prompt_builder.build_one_shot_prompt(
                **data_item,
                phase=phase,
                prompt_type=prompt_type,
                retrieve_content=retrieve_content,
                keep_row_order=keep_row_order,
            )
            few_shot_prompt_list.append(one_shot_prompt)
        if num_shots is not None:
            few_shot_prompt = '\n'.join(few_shot_prompt_list[:num_shots])
        else:
            few_shot_prompt = '\n'.join(few_shot_prompt_list[:self.args.num_shots])
        return few_shot_prompt

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            num_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:num_shots]
        few_shot_prompt_list[-1] = few_shot_prompt_list[-1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt = '\n'.join(few_shot_prompt_list)
        return few_shot_prompt

    def build_generate_prompt(
            self,
            data_item: Dict,
            phase: str,
            generate_type: Tuple,
            retrieve_content: bool = False,
            keep_row_order: bool = False,
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            phase=phase,
            generate_type=generate_type,
            retrieve_content=retrieve_content,
            keep_row_order=keep_row_order,
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple],
            phase: str,
            generate_type: Tuple,
            verbose: bool = False
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        result_idx_to_eid = []
        for p in prompts:
            result_idx_to_eid.extend([p[0]] * self.args.sampling_n)
        prompts = [p[1] for p in prompts]

        start_time = time.time()

        result = self._call_codex_api(
            engine=self.args.engine,
            prompt=prompts,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.sampling_n,
            stop=self.args.stop_tokens
        )
        print(f'Openai api one inference time: {time.time() - start_time}')

        if verbose:
            print('\n', '*' * 20, 'Codex API Call', '*' * 20)
            for prompt in prompts:
                print(prompt)
                print('\n')
            print('- - - - - - - - - - ->>')

        # parse api results
        response_dict = dict()
        for idx, g in enumerate(result['choices']):
            try:
                text = g['text']
                logprob = sum(g['logprobs']['token_logprobs'])
                if phase == 'generate':
                    if generate_type == ('nsql', 'question'):
                        g_nsql, g_question = text.split('Q:')
                        g_nsql, g_question = g_nsql.strip(), g_question.strip()
                    elif generate_type == ('question', 'nsql'):
                        g_question, g_nsql = text.split('NeuralSQL:')
                        g_question, g_nsql = g_question.strip(), g_nsql.strip()
                    elif generate_type == ('question', 'sql'):
                        g_question, g_nsql = text.split('SQL:')
                        g_question, g_nsql = g_question.strip(), g_nsql.strip()
                    elif generate_type == ('nsql',):
                        g_nsql, g_question = text, None
                    elif generate_type == ('sql',):
                        g_nsql, g_question = text, None
                    elif generate_type == ('answer',):
                        g_nsql, g_question = text, None
                    elif generate_type == ('chain of thought',):
                        g_cot = text.split('So the SQL answer is:')[0].strip()
                        g_nsql = text.split('So the SQL answer is:')[-1].strip()
                        # Normalize logprob by removing chain of thought text tokens
                        try:
                            tokens, token_logprobs = g['logprobs']['tokens'], g['logprobs']['token_logprobs']
                            nsql_start_tidx, nsql_end_tidx = 0, len(tokens)
                            for tidx in range(len(tokens)):
                                if tokens[tidx: tidx + 6] == \
                                        ['So', ' the', ' SQL', ' answer', ' is', ':']:
                                    nsql_start_tidx = tidx + 6
                                if tokens[tidx] in self.args.stop_tokens:
                                    nsql_end_tidx = tidx
                                    break
                            logprob = sum(token_logprobs[nsql_start_tidx: nsql_end_tidx])
                        except Exception as e:
                            logprob = -100
                            print(f"Find 'So the SQL answer is:' phrase fails: {e}\n"
                                  f"Set logprob=-100")
                    elif generate_type == ('chain of thought of qa',):
                        g_cot = text.split('So the answer is:')[0].strip()
                        g_nsql = text.split('So the answer is:')[-1].strip()
                        # Normalize logprob by removing chain of thought text tokens
                        try:
                            tokens, token_logprobs = g['logprobs']['tokens'], g['logprobs']['token_logprobs']
                            nsql_start_tidx, nsql_end_tidx = 0, len(tokens)
                            for tidx in range(len(tokens)):
                                if tokens[tidx: tidx + 5] == \
                                        ['So', ' the', ' answer', ' is', ':']:
                                    nsql_start_tidx = tidx + 5
                                if tokens[tidx] in self.args.stop_tokens:
                                    nsql_end_tidx = tidx
                                    break
                            logprob = sum(token_logprobs[nsql_start_tidx: nsql_end_tidx])
                        except Exception as e:
                            logprob = -100
                            print(f"Find 'So the answer is:' phrase fails: {e}\n"
                                  f"Set logprob=-100")
                    elif generate_type == ('npython',):
                        g_nsql, g_question = text, None
                    elif generate_type == ('python',):
                        g_nsql, g_question = text, None
                    else:
                        raise ValueError(f'generate type={generate_type} is not supported in phase={phase}')
                elif phase == 'filter':
                    if generate_type == ('nsql',):
                        g_nsql, g_question = text, None
                    elif generate_type == ('sql',):
                        g_nsql, g_question = text, None
                    elif generate_type == ('question',):
                        g_question, g_nsql = text, None
                    else:
                        raise ValueError(f'generate type={generate_type} is not supported in phase={phase}')

                eid = result_idx_to_eid[idx]
                eid_pairs = response_dict.get(eid, None)
                if eid_pairs is None:
                    eid_pairs = []
                    response_dict[eid] = eid_pairs

                # TODO: Wrap return values into a Class
                if generate_type in [('chain of thought',), ('chain of thought of qa',)]:
                    eid_pairs.append((g_nsql, g_cot, logprob))
                else:
                    eid_pairs.append((g_nsql, g_question, logprob))

                if verbose:
                    print(text)

            except ValueError as e:
                if verbose:
                    print('----------- Error Msg--------')
                    print(e)
                    print(text)
                    print('-----------------------------')
                pass

        return response_dict

    # @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIConnectionError))
    def _call_codex_api(
            self,
            engine: str,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str]
    ):
        start_time = time.time()
        result = None
        while result is None:
            try:
                key = self.keys[self.current_key_id]
                self.current_key_id = (self.current_key_id + 1) % len(self.keys)
                print(f"Using openai api key: {key}")
                result = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    api_key=key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    logprobs=1
                )
                print('openai api inference time:', time.time() - start_time)
                return result
            except Exception as e:
                print(e, 'Retry.')
                time.sleep(5)
