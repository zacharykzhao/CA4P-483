from prompt_template import prompte_template_1
from tqdm import tqdm
import json
import os

TAR_FIELD = ["e_21", "e_22", "e_23", "e_24", "e_25", "e_26", "e_27"]

ner_label_map = {
    "e_21": "data",
    "e_22": "collection",
    "e_23": "sharing",
    "e_24": "controller",
    "e_25": "condition",
    "e_26": "receiver",
    "e_27": "purpose"
}

Punct = [" ", "，", "：", "。", "｜", "/", "、", "！", "～", "#", "¥", "%",
         "！", "？", "｡", "＂", "＃", "＄", "％", "＆", "＇", "（", "）",
         "＊", "＋", "，", "－", "／", "：", "；", "＜", "＝", "＞", "＠",
         "［", "＼", "］", "＾", "＿", "｀", "｛", "｜", "｝", "～", "｟",
         "｠", "｢", "｣", "､", "、", "〃", "》", "「", "」", "『", "』",
         "【", "】", "〔", "〕", "〖", "〗", "〘", "〙", "〚", "〛", "〜",
         "〝", "〞", "〟", "〰", "〾", "〿", "–", "—", "‘", "’", "‛", "“",
         "”", "„", "‟", "…", "‧", "﹏", "."]


def get_ner_label(tagtog_label: str):
    return ner_label_map[tagtog_label]


# print(prompte_template_1.format(pp_sentence="test"))

def get_NER(tar_folder):
    data_all = []
    ner_all = []
    for cur_file in tqdm(os.listdir(tar_folder)):
        file_name = tar_folder + '/' + cur_file

        f = open(file_name, 'r', encoding="utf8")
        data = json.load(f)
        tmp = data[1:]

        for i, data_i in enumerate(data):
            if data_i['classId'] in TAR_FIELD:
                # 当前为NER 标注，跳过
                continue
            idx_start = data_i['start']
            idx_end = data_i['end']
            cur_sen = data_i['text']
            cur_ner = ['o' for it in cur_sen]
            flag_1 = False

            rmv_idx = []
            output_ner = ""
            for z1, dt in enumerate(tmp):

                data_tmp = tmp[z1]

                if data_tmp['start'] >= idx_start and data_tmp['end'] <= idx_end \
                        and data_tmp['classId'] in TAR_FIELD:
                    # 标记的数据在当前句子中，处理一下
                    flag_1 = True
                    sen_idx_start = data_tmp['start'] - idx_start
                    sen_idx_end = sen_idx_start + data_tmp['end'] - data_tmp['start']
                    output_ner = output_ner+f"【{data_tmp['text']}|{get_ner_label(data_tmp['classId'])}】, "
            output_ner = output_ner[:-1]

            if flag_1 is True and len(output_ner)>0:
                # 该句子、段落有标记
                data_all.append({
                    "sentence":cur_sen,
                    "output":output_ner
                })
        f.close()
    return data_all


if __name__ == '__main__':
    final_data = get_NER("../annotation_data")
    print(len(final_data))
    #
    f_raw = open("./llmpp_raw_data.jsonl", 'w', encoding='utf-8')
    for one_data in final_data:
        f_raw.write(json.dumps(one_data, ensure_ascii=False)+'\n')
    f_raw.close()
    #
    f_prompt = open("./llmpp_prompts.jsonl", "w", encoding="utf-8")
    for one_data in final_data:
        one_data["prompt"] = prompte_template_1.format(pp_sentence=one_data["sentence"])
        f_prompt.write(json.dumps(one_data, ensure_ascii=False) + '\n')
    f_prompt.close()
