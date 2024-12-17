import json
from metrics_utils import calculate_metrics_with_confusion_matrix
import numpy as np

results_file = "./gpt35_results.jsonl"

label_map = {
    "data": 1,
    "controller": 2,
    "collection": 3,
    "sharing": 4,
    "condition": 5,
    "purpose": 6,
    "receiver": 7
}

int2label = [
    "O", "Data", "Controller", "Collection", "Sharing", "Condition", "Purpose", "Receiver"
]

with open(results_file, 'r', encoding='utf-8') as f:
    results = [json.loads(one_data) for one_data in f.readlines()]


def parse_result(data_str: str, error_num):
    data_str = data_str.replace("，", ",").replace("｜", "|")
    data = data_str.split(",")
    cur_results = {}
    for one_data in data:
        if len(one_data) < 1:
            error_num = error_num + 1
            continue
        one_data = one_data.strip().lstrip("【").rstrip("】")
        split_one_data = one_data.split("|")
        if len(split_one_data) == 1:
            continue
        entity_name, entity_type = split_one_data[0], split_one_data[1]
        if entity_type not in cur_results:
            cur_results[entity_type] = []
        cur_results[entity_type].append(entity_name)
    return cur_results, wrong_output_num


def get_metrics(gt, pred, confusion_list):
    for entity_type in gt:
        for entity_name in gt[entity_type]:
            confusion_list['ground_truth'].append(label_map[entity_type])
            if entity_type not in pred:
                confusion_list['prediction'].append(0)
            else:
                PRED_MATCH = False
                for pred_en in pred[entity_type]:
                    if pred_en == entity_name:
                        confusion_list['prediction'].append(label_map[entity_type])
                        PRED_MATCH = True
                        break  # in case there are llm hallucination
                if not PRED_MATCH:
                    confusion_list['prediction'].append(0)


if __name__ == '__main__':
    final_labels = []
    wrong_output_num = 0
    metrics = {
        "ground_truth": [],
        "prediction": []
    }
    pred_metrics = {}
    for result in results:
        label = result['output']
        prediction = result['model_output']
        duration = result['duration']
        label, _ = parse_result(label, 0)
        prediction, wrong_output_num = parse_result(prediction, wrong_output_num)
        #
        get_metrics(label, prediction, metrics)

    #
    class_metrics, overall_metrics, confusion_matrix, classes = calculate_metrics_with_confusion_matrix(
        metrics['prediction'], metrics['ground_truth'])
    print("Confusion Matrix:")
    print(confusion_matrix)

    print("\nPer-Class Metrics:")
    for cls, metrics in class_metrics.items():
        print(f"Class {int2label[cls]}: {metrics}")

    print("\nOverall Metrics:")
    print(overall_metrics)

    print(f"{wrong_output_num} wrong output format")
