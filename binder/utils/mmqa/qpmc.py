import os
import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")


class Question_Passage_Match_Classifier(object):
    """result are from a T5-3b model finetuned on train set of MMQA."""

    def __init__(self):
        self.qa_pairs_should_retrieve = None
        self.load_retrieve_info()

    def load_retrieve_info(self):
        df = pd.read_csv(os.path.join(ROOT_DIR, "utils", "mmqa", "qpmc_mmqa_dev.csv"))
        qa_pairs_should_retrieve = {}
        for index, row in df.iterrows():
            qa = row['question'].lower()
            prediction = row['prediction']
            qa_pairs_should_retrieve[qa] = True if prediction == "['yes']" else False
        self.qa_pairs_should_retrieve = qa_pairs_should_retrieve

    def judge_match(self, question, passage):
        return self.qa_pairs_should_retrieve['qa: {} \n {}'.format(question.lower(), passage.lower())]