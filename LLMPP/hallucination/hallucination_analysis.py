import json
import os.path

label_map = {
    "data": 1,
    "controller": 2,
    "collection": 3,
    "sharing": 4,
    "condition": 5,
    "purpose": 6,
    "receiver": 7
}




def read_file(target_file):
    data = []
    f = open(target_file, 'r', encoding='utf-8', errors='replace')
    for one_data in f.readlines():
        try:
            data.append(json.loads(one_data))
        except Exception:
            # print(one_data)
            continue
    # with open() as f:
    #     data = [json.loads(one_data) for one_data in f.readlines()]
    return data


def parse_results_few_shot(data_str):
    if isinstance(data_str, dict):
        data_str = data_str['content']
    data_str = data_str.replace("，", ",").replace("｜", "|").replace("、", ",").replace("：", ":").lower()
    cur_results = {}
    if "\n" in data_str:
        """deal with format with multiple lines"""
        data = data_str.split("\n")
        for one_data in data:
            if ":" in one_data:
                """deal with format:
                entity_type: entities"""
                data_split = one_data.split(":")
                entity_type = data_split[0]
                for et in label_map:
                    if et in entity_type:
                        if et not in cur_results:
                            cur_results[et] = []
                        cur_results[et].extend(list(set(data_split[1].split(','))))
            else:
                """deal with the format that each line with 【entities｜entity type】"""
                temp_results = one_data.split(",")
                for one_temp in temp_results:
                    one_temp = one_temp.strip().lstrip("【").rstrip("】")
                    split_temp = one_data.split("|")
                    if len(split_temp) == 1: continue
                    entity_name_temp, entity_type_temp = split_temp[0], split_temp[1]
                    for et in label_map:
                        if et in entity_type_temp:
                            if et not in cur_results:
                                cur_results[et] = []
                            cur_results[et].append(entity_name_temp)
    else:
        """deal with format: entity|type, **entity|type**"""
        data = data_str.split(",")
        for one_data in data:
            one_data = one_data.strip().lstrip("*").rstrip("*")
            split_one_data = one_data.split("|")
            if len(split_one_data) == 1:
                continue
            entity_name, entity_type = split_one_data[0], split_one_data[1]
            for et in label_map:
                if et in entity_type:
                    if et not in cur_results:
                        cur_results[et] = []
                    cur_results[et].append(entity_name)
    return cur_results, -1


def parse_result(data_str, error_num):
    if isinstance(data_str, dict):
        data_str = data_str['content']
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
    return cur_results


def analyze_hallucination(data_str, sentence, metrics):
    if isinstance(data_str, dict):
        data_str = data_str['content']
    data_str = data_str.replace("，", ",").replace("｜", "|")
    data = data_str.split(",")
    for one_data in data:
        if len(one_data) < 1: continue
        one_data = one_data.strip().lstrip("【").rstrip("】")
        split_one_data = one_data.split("|")
        if len(split_one_data) == 1:
            continue
        entity_name, entity_type = split_one_data[0], split_one_data[1]
        metrics['G_e'] += 1
        metrics['G_t'] += 1
        if entity_name not in sentence:
            metrics['I_e'] += 1
        if entity_type.lower() not in label_map:
            metrics['I_t'] += 1


if __name__ == '__main__':
    models = ['gpt', 'llama', 'qwen']
    prompt_engineerings = ["assigning_a_role",
                           "task_description",
                           "context_aware",
                           "instruction_tuned",
                           "few_shot"]

    for model in models:
        results_folder = os.path.join("../ablation", model)
        for pe in prompt_engineerings:
            print(f"{'=' * 20} \n {model},\t{pe} \n")
            results_file = os.path.join(results_folder, f"{pe}.jsonl")
            results = read_file(results_file)
            final_metrics = {
                "I_e": 0, "G_e": 0, "I_t": 0, "G_t": 0
            }
            metrics = {
                "ground_truth": [],
                "prediction": []
            }
            for result in results:
                label = result['output']
                prediction = result['model_output']
                duration = result['duration']
                pp_sentence = result['sentence']
                analyze_hallucination(prediction, pp_sentence, final_metrics)
            H_e = final_metrics['I_e']/final_metrics['G_e']
            H_t = final_metrics['I_t']/final_metrics['G_t']
            print("Hallucination_entity, Hallucination_type")
            print(f"{H_e*100:.2f}%, {H_t*100:.2f}%")

    for model in ['gpt35','llama','qwen']:

        print(f"\n{model} main results")
        results_file = os.path.join("../main_results", f"{model}_results.jsonl")
        results = read_file(results_file)
        final_metrics = {
            "I_e": 0, "G_e": 0, "I_t": 0, "G_t": 0
        }
        metrics = {
            "ground_truth": [],
            "prediction": []
        }
        for result in results:
            label = result['output']
            prediction = result['model_output']
            duration = result['duration']
            pp_sentence = result['sentence']
            analyze_hallucination(prediction, pp_sentence, final_metrics)
        H_e = final_metrics['I_e']/final_metrics['G_e']
        H_t = final_metrics['I_t']/final_metrics['G_t']
        print("Hallucination_entity, Hallucination_type")
        print(f"{H_e*100:.2f}%, {H_t*100:.2f}%")