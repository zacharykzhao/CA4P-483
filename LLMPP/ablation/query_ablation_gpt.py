import requests
import json
import time
from tqdm import tqdm
from prompt_templates import prompt_template_format
from prompt_templates import prompt_eng_json
import os


def query_gpt35(prompt):
    url = "https://oa.api2d.net/v1/chat/completions"

    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "safe_mode": False
    })
    headers = {
        'Authorization': 'Bearer yourapikey',
        'Content-Type': 'application/json'
    }

    beg_time = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    duration = time.time() - beg_time
    response_content = json.loads(response.text)
    if "choices" not in response_content:
        return "", duration
    response_content = response_content['choices'][0]['message']['content']
    print(f"Duration:{duration} ::: {response_content}")
    return response_content, duration


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as file:
        datatmp = [json.loads(line) for line in file]
    return datatmp


if __name__ == '__main__':
    json_path = "llmpp_raw_data.jsonl"
    save_path = "./ablation"
    for pe in prompt_eng_json:
        prompt_template = ""
        print(f"===== {pe}")
        # construct prompt
        for pe_t in prompt_eng_json:
            if not pe_t == pe:
                prompt_template += prompt_eng_json[f"{pe_t}"]
        prompt_template += prompt_template_format

        # print(prompt_template)
        save_file = os.path.join(save_path, f"{pe}.jsonl")

        results = []
        existing_lines = 0
        if os.path.exists(save_file):
            existing_data = open(save_file, 'r')
            existing_lines = len(existing_data.readlines())
            existing_data.close()
        wf = open(save_file, 'a', encoding='utf-8')

        data = load_data(json_path)
        count = 0
        for one_data in tqdm(data):
            if count < existing_lines:
                count = count + 1
                continue
            sentence = one_data["sentence"]
            prompt = prompt_template.replace("{pp_sentence}", sentence)

            # print(prompt)
            messages = [
                {"role": "user", "content" : prompt}
            ]
            try:
                start_time = time.time()
                resp, duration = query_gpt35(prompt)
                duration = time.time() - start_time
            except Exception:
                time.sleep(5)
                start_time = time.time()
                resp, duration = query_gpt35(prompt)
                duration = time.time() - start_time
            model_output = resp
            one_data['prompt'] = prompt
            one_data['duration'] = duration
            one_data['model_output'] = model_output
            print(f"\nDuration {duration} \t\t {model_output}")
            results.append(one_data)
            wf.write(json.dumps(one_data, ensure_ascii=False) + '\n')
            # time.sleep(1)
