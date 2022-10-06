import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

# wikitq nsql annotation command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset wikitq \
# --dataset_split test \
# --prompt_file templates/prompts/prompt_wikitq_v3.txt \
# --n_parallel_prompts 1 \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 20 \
# -v""")

# wikitq nsql execution command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset wikitq \
# --dataset_split test \
# --qa_retrieve_pool_file templates/qa_retrieve_pool.json \
# --input_program_file binder_program_wikitq_test.json \
# --output_program_execution_file binder_program_wikitq_test_exec.json \
# --vote_method simple
# """)

# tab_fact nsql annotation command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset tab_fact \
--dataset_split test \
--prompt_file templates/prompts/prompt_tab_fact_sqllike_v3.txt \
--n_parallel_prompts 1 \
--n_processes 1 \
--n_shots 18 \
--max_generation_tokens 256 \
--temperature 0.6 \
--sampling_n 50 \
-v""")

# tab_fact nsql execution command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset tab_fact \
# --dataset_split test \
# --qa_retrieve_pool_file templates/qa_retrieve_pool.json \
# --input_program_file binder_program_tab_fact_test.json \
# --output_program_execution_file binder_program_tab_fact_test_exec.json \
# --vote_method answer_biased \
# --answer_biased 1 \
# --answer_biased_weight 4 \
# """)

# More
# tab_fact nsql annotation command

# mmqa nsql annotation command

# mmqa nsql execution command

# Analysis(Still working on cleaning these code)
# scalability command
# robustness command
# python command