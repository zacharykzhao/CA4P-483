import json
from metrics_utils import calculate_metrics_with_confusion_matrix, plot_confusion_matrix_seaborn
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

results_file = "./qwen_results.jsonl"

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

wrong_output_num = 0

with open(results_file, 'r', encoding='utf-8') as f:
    results = [json.loads(one_data) for one_data in f.readlines()]


def parse_result(data_str: str, error_num):
    if isinstance(data_str, dict):
        data_str = data_str['content']
    data_str = data_str.replace("，", ",").replace("｜", "|")
    data = data_str.split(",")
    cur_results = {}
    for one_data in data:
        if len(one_data) < 1:
            error_num += 1
            continue
        one_data = one_data.strip().lstrip("【").rstrip("】")
        split_one_data = one_data.split("|")
        if len(split_one_data) == 1:
            continue
        entity_name, entity_type = split_one_data[0], split_one_data[1]
        if entity_type not in cur_results:
            cur_results[entity_type] = []
        cur_results[entity_type].append(entity_name)
    return cur_results, error_num


def get_metrics(gt, pred, confusion_list):
    for entity_type in gt:
        for entity_name in gt[entity_type]:
            confusion_list['ground_truth'].append(label_map[entity_type])
            PRED_MATCH = False
            #
            for pred_type in pred:
                if PRED_MATCH:
                    break
                if pred_type not in label_map:
                    continue
                for pred_en in pred[pred_type]:
                    if pred_en == entity_name:
                        confusion_list['prediction'].append(label_map[pred_type])
                        PRED_MATCH = True
                        break
                    elif len(entity_name) > 5 or len(pred_en) > 5:
                        if entity_name in pred_en or pred_en in entity_name:
                            confusion_list['prediction'].append(label_map[pred_type])
                            PRED_MATCH = True
                            break
            if not PRED_MATCH:
                confusion_list['prediction'].append(0)




if __name__ == '__main__':

    final_labels = []
    metrics = {
        "ground_truth": [],
        "prediction": []
    }
    wrong_output_num = 0
    pred_metrics = {}
    for result in results:
        label = result['output']
        prediction = result['model_output']
        duration = result['duration']
        label, _ = parse_result(label,0)
        prediction, wrong_output_num = parse_result(prediction, wrong_output_num)
        #
        get_metrics(label, prediction, metrics)

    #
    class_metrics, overall_metrics, confusion_matrix, classes = calculate_metrics_with_confusion_matrix(
        metrics['prediction'], metrics['ground_truth'])
    print("Confusion Matrix:")
    print(confusion_matrix)

    print("\nPer-Class Metrics:")
    for cls, results in class_metrics.items():
        # print(f"Class {int2label[cls]}: {metrics}")
        print(f"{int2label[cls]},{results['Precision']},"
              f"{results['Recall']},{results['F1']},"
              f"{results['TP']},{results['FP']},"
              f"{results['FN']},{results['TN']}")

    print("\nOverall Metrics:")
    print(overall_metrics)
    print(f"{overall_metrics['Precision']},"
          f"{overall_metrics['Recall']},"
          f"{overall_metrics['F1']},"
          f"{overall_metrics['TP']},{overall_metrics['FP']},"
          f"{overall_metrics['FN']},{overall_metrics['TN']},")