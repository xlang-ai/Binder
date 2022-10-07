import json
import os
import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")


class Question_Image_Match_Classifier(object):
    """result are from a T5-3b model finetuned on train set of MMQA."""

    def __init__(self):
        self.whether_retrieve_image = None
        self.qi_pairs_should_retrieve = None
        self.load_retrieve_info()
        self.caption_info = None
        with open(os.path.join(ROOT_DIR, "utils", "mmqa", "mmqa_captions.json"), "r") as f:
            self.caption_info = json.load(f)

    def load_retrieve_info(self):
        df_qc = pd.read_csv(os.path.join(ROOT_DIR, "utils", "mmqa", "qc_mmqa_dev.csv"))
        whether_retrieve_image = {}
        for index, row in df_qc.iterrows():
            _id = row['id']
            prediction = row['prediction']
            whether_retrieve_image[_id] = True if prediction == "['yes']" else False
        self.whether_retrieve_image = whether_retrieve_image

        df_qimc = pd.read_csv(os.path.join(ROOT_DIR, "utils", "mmqa", "qimc_mmqa_dev.csv"))
        qi_pairs_should_retrieve = {}
        for index, row in df_qimc.iterrows():
            qa = row['question'].lower()
            prediction = row['prediction']
            qi_pairs_should_retrieve[qa] = True if prediction == "['yes']" else False
        self.qi_pairs_should_retrieve = qi_pairs_should_retrieve

    def judge_match(self, _id, question, pic):
        # fixme: hardcode since it is done in pipeline, change that in the future
        if not self.whether_retrieve_image[_id]:
            return False
        image_caption = self.caption_info[os.path.split(pic)[-1].split(".")[0]]
        return self.qi_pairs_should_retrieve['qa: {} \n{}'.format(question.lower(), image_caption.lower())]