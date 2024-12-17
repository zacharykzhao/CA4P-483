import requests
import json
import time
from tqdm import tqdm


def query_claude(prompt):
    url = "https://oa.api2d.net/claude/v1/messages"

    payload = json.dumps({
        "model": "claude-3-opus-20240229",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 512
    })
    headers = {
        'Authorization': 'Bearer fk229918-z2E3i5ZhLzq7bgNQ4EFQBbbtz4UGXqRI',
        'Content-Type': 'application/json'
    }

    beg_time = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    dura = time.time() - beg_time
    ra = json.loads(response.text)
    model_output = ra["content"][0]['text']
    print(f"Duration:{dura} ::: {model_output}")

    return model_output, dura


if __name__ == '__main__':
    json_path = "llmpp_prompts.jsonl"
    save_file = "claude3_results.jsonl"
    fs = open(save_file, 'a', encoding='utf-8')

    with open(json_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    for one_data in tqdm(data):
        prompt = one_data['prompt']
        resp, duration = query_claude(prompt)
        one_data['duration'] = duration
        one_data['model_output'] = resp
        fs.write(json.dumps(one_data, ensure_ascii=False) + '\n')
