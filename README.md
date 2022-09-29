# Binder

## Dependencies
To establish the environment run this code in the shell:
```bash
conda env create -f py3.7binder.yaml
pip install records==0.5.3
```
That will create the environment `binder` we used.


## Usage

### Environment setup
Activate the environment by running
``````shell
conda activate binder
``````

### Environment Variable Set Up
It's better to set environment variable to `TOKENIZERS_PARALLELISM` to `false`, if you use our multiprocess script to run code.

### Add key
Get `private key`(sk-xxxx like) from OpenAI, and save the key in `key.txt` file, make sure you have the rights to access the model you need.

### Run
Take running WikiTableQuestion as an example, first cd into its directory `scripts`. (For TabFact and MultiModalQA, they are `scripts_tab_fact` and `scripts_mmqa` respectively.)

Then run the script of each setting for End2end QA, SQL and NSQL(SQL Binder). 
```bash
python multiprocess_annotate_fixprompt_NSQL.py
```

After the annotation generation, feed the file name of the generated file into the execution script to perform execution on it.
```bash
python multiprocess_execute_NSQL.py
```
