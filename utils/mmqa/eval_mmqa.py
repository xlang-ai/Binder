from utils.mmqa.evaluator import evaluate_predictions
import json

ANNOTATION_RESULT_PATH = "../results_mmqa/unfiltered_mmqa_nsqls_mmqa_v2_all_standard.json"
EXECUTION_RESULT_PATH = "../results_mmqa/unfiltered_mmqa_nsqls_mmqa_v2_all_standard_new_qa_pool_bug_fixed_v1.jsonn"

if __name__ == '__main__':
    right = 0
    all_num = 0
    with open(ANNOTATION_RESULT_PATH, "r") as f:
        all_data = json.load(f)
    with open(EXECUTION_RESULT_PATH, "r") as f:
        pred_data = json.load(f)

    pred_dict = {eid: [str(a) for a in pred_data[eid]['pred_answer']] for eid in pred_data}
    gold_dict = {eid: [str(a) for a in pred_data[eid]['gold_answer']] for eid in pred_data}
    eval_scores, instance_eval_results = evaluate_predictions(pred_dict, gold_dict)
    print(eval_scores)
    print(instance_eval_results)