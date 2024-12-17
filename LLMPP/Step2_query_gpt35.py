import requests
import json
import time
from tqdm import tqdm


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
        'Authorization': 'Bearer **********',
        'Content-Type': 'application/json'
    }

    beg_time = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    duration = time.time() - beg_time
    response_content = json.loads(response.text)['choices'][0]['message']['content']
    print(f"Duration:{duration} ::: {response_content}")
    return response_content, duration


if __name__ == '__main__':
    json_path = "llmpp_prompts.jsonl"
    save_file = "gpt35_results.jsonl"
    fs = open(save_file, 'a', encoding='utf-8')

    with open(json_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    for idx, one_data in tqdm(enumerate(data)):
        if idx < 39:
            continue
        prompt = one_data['prompt']
        resp, duration = query_gpt35(prompt)
        one_data['duration'] = duration
        one_data['model_output'] = resp
        fs.write(json.dumps(one_data, ensure_ascii=False) + '\n')
