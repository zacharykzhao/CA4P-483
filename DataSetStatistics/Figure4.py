import json
import os
import matplotlib.pyplot as plt

ner_label_map = {
    "e_21": "data",
    "e_22": "collect",
    "e_23": "share",
    "e_24": "handler",
    "e_25": "condition",
    "e_26": "subjects",
    "e_27": "purpose"
}


def init_result(ENTITY_FIELD):
    results = {}
    for i in ENTITY_FIELD:
        results[i] = []
    return results


if __name__ == '__main__':
    json_dataset = './CA4P-483/data_preprocess/merged_json'

    comp_dis = init_result(ner_label_map)

    for p in os.listdir(json_dataset):
        if p.endswith('DS_Store'):
            continue
        file_name = os.path.join(json_dataset, p)
        # print(file_name)
        f = open(file_name, 'r', encoding='utf8')
        data = json.load(f)
        tmp = data
        tmp_results = {}
        for tid in ner_label_map:
            tmp_results[tid] = 0
        for i, data_i in enumerate(data):
            id = data_i['classId']

            if id in ner_label_map:  # if_2
                tmp_results[id] += 1
        for tid in tmp_results:
            comp_dis[tid].append(tmp_results[tid])

    all_data = []
    xt = []
    for i in ner_label_map:
        all_data.append(comp_dis[i])
        xt.append(ner_label_map[i])
    plt.boxplot(all_data, notch=True)
    plt.grid()
    plt.xticks(range(1, 8), xt)
    plt.show()
