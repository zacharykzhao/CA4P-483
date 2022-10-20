# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2022/10/20



import os
import json

from data_preprocess.util_kz import cut_sentz, get_ner_label
from tqdm import tqdm

TAR_FIELD = ["e_21", "e_22", "e_23", "e_24", "e_25", "e_26", "e_27"]
PUNCTIONs = [" ", "，", "：", "。", "｜", "/", "、", "！", "～", "#", "¥", "%"]
Punct = [" ", "，", "：", "。", "｜", "/", "、", "！", "～", "#", "¥", "%",
         "！", "？", "｡", "＂", "＃", "＄", "％", "＆", "＇", "（", "）",
         "＊", "＋", "，", "－", "／", "：", "；", "＜", "＝", "＞", "＠",
         "［", "＼", "］", "＾", "＿", "｀", "｛", "｜", "｝", "～", "｟",
         "｠", "｢", "｣", "､", "、", "〃", "》", "「", "」", "『", "』",
         "【", "】", "〔", "〕", "〖", "〗", "〘", "〙", "〚", "〛", "〜",
         "〝", "〞", "〟", "〰", "〾", "〿", "–", "—", "‘", "’", "‛", "“",
         "”", "„", "‟", "…", "‧", "﹏", "."]


def get_NER(tar_folder, save_file="tmp.txt"):
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
            for z1, dt in enumerate(tmp):

                data_tmp = tmp[z1]
                if data_tmp['start'] > idx_start and data_tmp['end'] < idx_end \
                        and data_tmp['classId'] in TAR_FIELD:
                    # 标记的数据在当前句子中，处理一下
                    flag_1 = True
                    sen_idx_start = data_tmp['start'] - idx_start
                    sen_idx_end = sen_idx_start + data_tmp['end'] - data_tmp['start']
                    for ner_i in range(sen_idx_start, sen_idx_end):
                        # 判断标点符号
                        if cur_sen[ner_i] in Punct:
                            continue
                        # BME label
                        if ner_i == sen_idx_start:
                            cur_ner[ner_i] = "B-" + get_ner_label(data_tmp['classId'])
                        elif ner_i == sen_idx_end - 1:
                            cur_ner[ner_i] = "E-" + get_ner_label(data_tmp['classId'])
                        else:
                            cur_ner[ner_i] = "M-" + get_ner_label(data_tmp['classId'])
                        # cur_ner[ner_i] = data_tmp['classId']

                    rmv_idx.append(z1)
            rmv_idx.reverse()
            for del_i in rmv_idx:  # 删掉用过的元素 节省时间
                del tmp[del_i]
            rmv_idx = []
            if flag_1 is True:
                # 该句子、段落有标记
                data_all.extend(cur_sen)
                ner_all.extend(cur_ner)

    f.close()

    f = open(save_file, 'w', encoding='utf-8')
    for i in range(len(data_all)):
        if data_all[i] in ["\n", " ", "\t", "|", "|", "*"] and ner_all[i] == "o":  # deal with space or table
            # f.write("\n")
            continue

        if data_all[i] in ["。"]:
            f.write("\n")
            continue
        f.write(data_all[i] + '\t' + ner_all[i] + '\n')


if __name__ == '__main__':
    get_NER("annotation_data", save_file="all_all.txt")
