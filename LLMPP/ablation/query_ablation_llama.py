import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time
from tqdm import tqdm 
import os
from prompt_templates import prompt_template_format
from prompt_templates import prompt_eng_json

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def load_data(data_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data


if __name__=="__main__":
    json_path = "./llmpp_raw_data.jsonl"
    save_path = "./ablation/llama"
    
    
    model_path = "./models/Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda:3",
        max_new_tokens=512
    )
    

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
        wf = open(save_file, 'a', encoding='utf-8')

        data = load_data(json_path)

        for one_data in tqdm(data):
            sentence = one_data["sentence"]
            prompt = prompt_template.replace("{pp_sentence}", sentence)

            # print(prompt)
            messages = [
                {"role": "user", "content": prompt}
            ]
            start_time = time.time()
            outputs = pipeline(
                messages
            )
            one_data['prompt'] = prompt
            duration = time.time() - start_time
            model_output = outputs[0]["generated_text"][-1]
            one_data['duration'] = duration
            one_data['model_output'] = model_output
            print(f"\nDuration {duration} \t\t {model_output}")
            results.append(one_data)
            wf.write(json.dumps(one_data, ensure_ascii=False)+'\n')

