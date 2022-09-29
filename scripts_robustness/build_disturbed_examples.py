import os
import argparse
import random
import pandas as pd
import re

random.seed(42)
pd.set_option('display.max_columns', None)


def main():
    # Load
    examples = pd.read_csv(os.path.join(args.data_dir, 'data/', args.input_example_file), sep='\t')
    disturb_examples = pd.DataFrame(data=[], columns=examples.columns)
    candidate_disturb_table_ids = []
    for table_type in ['200-csv', '201-csv', '202-csv', '203-csv', '204-csv']:
        table_file_list = os.listdir(os.path.join(args.data_dir, 'disturbed_csv/', table_type))
        candidate_disturb_table_ids.extend([f'csv/{table_type}/{table_file}' for table_file in table_file_list])
    # Iterate all examples and keep the first 150
    for row_id, row in examples.iterrows():
        disturb_table_id = row['context'][:-4] + '_' + row['id'] + '.tsv'
        if disturb_table_id in candidate_disturb_table_ids:
            row['context'] = disturb_table_id
            disturb_examples = disturb_examples.append(row, ignore_index=True)
            NUMBER_PATTERN = re.compile('[+-]?([0-9]*[.])?[0-9]+')
            if re.findall(NUMBER_PATTERN, row['utterance']):
                print(f"{disturb_table_id}")
        if len(disturb_examples) == 150:
            break
    print(disturb_examples)

    # # Save
    # disturb_examples.to_csv(os.path.join(args.data_dir, 'data/', args.output_example_file), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file path or name
    parser.add_argument('--data_dir', type=str,
                        default='../datasets/data/downloads/extracted/711cd80f2d751dffbf163586e177e0f1c063b3d7220aa286cd7630c8d70cd06c/WikiTableQuestions-master/')
    parser.add_argument('--input_example_file', type=str, default='random-split-1-dev_remove_line2532.tsv')
    parser.add_argument('--output_example_file', type=str, default='random-split-1-dev_robustness_ablation.tsv')
    args = parser.parse_args()

    main()
