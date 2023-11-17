# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2022/5/24
# System: linux

import os
from tqdm import tqdm
import json
import numpy as np

Punct = [" ", "，", "：", "。", "｜", "/", "、", "！", "～", "#", "¥", "%",
         "！", "？", "｡", "＂", "＃", "＄", "％", "＆", "＇", "（", "）",
         "＊", "＋", "，", "－", "／", "：", "；", "＜", "＝", "＞", "＠",
         "［", "＼", "］", "＾", "＿", "｀", "｛", "｜", "｝", "～", "｟",
         "｠", "｢", "｣", "､", "、", "〃", "》", "「", "」", "『", "』",
         "【", "】", "〔", "〕", "〖", "〗", "〘", "〙", "〚", "〛", "〜",
         "〝", "〞", "〟", "〰", "〾", "〿", "–", "—", "‘", "’", "‛", "“",
         "”", "„", "‟", "…", "‧", "﹏", "."]

ner_label_map = {
    "e_21": "data",
    "e_22": "collect",
    "e_23": "share",
    "e_24": "handler",
    "e_25": "condition",
    "e_26": "subjects",
    "e_27": "purpose"
}
ENTITY_FIELD = ['e_21', 'e_22', 'e_23', 'e_24', 'e_25', 'e_26', 'e_27']
TAR_FIELD = ['e_21', 'e_22', 'e_23', 'e_24', 'e_26']


def read_BIOS_file(train_file):
    f = open(train_file, 'r')
    context = f.readlines()
    f.close()
    data = {}
    for i in context:
        if i.startswith('\n'):
            continue
        tmp = i.replace('\n', '').split('\t')[1]
        if tmp.startswith('B'):
            entity_type = tmp.split('-')[1]
            if entity_type not in data:
                data[entity_type] = 0
            data[entity_type] += 1
    return data


def get_train_test_dev_info(BIOS_path):
    train_file = os.path.join(BIOS_path, 'train.char.bios')
    test_file = os.path.join(BIOS_path, 'test.char.bios')
    dev_file = os.path.join(BIOS_path, 'dev.char.bios')
    train = read_BIOS_file(train_file)
    test = read_BIOS_file(test_file)
    dev = read_BIOS_file(dev_file)
    return train, test, dev


def init_result(ENTITY_FIELD):
    results = {}
    for i in ENTITY_FIELD:
        results[i] = []
    return results


if __name__ == '__main__':
    json_dataset = './CA4P-483/data_preprocess/merged_json'
    BIOS_dataset = './CA4P-483/train_dev_test'
    #

    results_static = init_result(ENTITY_FIELD)
    sent_all = 0
    sent_with_entity = 0
    sent_we = []
    sent_z = []
    char_num = 0
    #
    for p in os.listdir(json_dataset):
        if p.endswith('DS_Store'):
            continue
        file_name = os.path.join(json_dataset, p)
        # print(file_name)
        f = open(file_name, 'r', encoding='utf8')
        data = json.load(f)
        tmp = data

        for i, data_i in enumerate(data):
            id = data_i['classId']

            if id in ENTITY_FIELD:  # if_2
                # 当前为NER 标注，跳过
                continue
            # end if_2
            idx_start = data_i['start']
            idx_end = data_i['end']
            cur_sen = data_i['text']
            cur_ner = ['o' for it in cur_sen]
            flag_1 = False
            #
            rmv_idx = []
            for z1, dt in enumerate(tmp):
                data_tmp = tmp[z1]
                if data_tmp['start'] > idx_start and data_tmp['end'] < idx_end \
                        and data_tmp['classId'] in ENTITY_FIELD:
                    # 标记的数据在当前句子中，处理一下
                    # results_static[data_tmp['classId']].append() # 统计+1
                    flag_1 = True
                    sen_idx_start = data_tmp['start'] - idx_start
                    sen_idx_end = sen_idx_start + data_tmp['end'] - data_tmp['start']
                    #
                    if data_tmp['classId'] in TAR_FIELD:
                        tmp1 = []
                        zt = ''
                        for ner_i in range(sen_idx_start, sen_idx_end):
                            # 判断标点符号
                            if cur_sen[ner_i] in Punct:
                                # results_static[data_tmp['classId']] += 1
                                tmp1.append(zt)
                                zt = ''
                                continue
                            # BME label
                            zt += data_tmp['text'][ner_i - sen_idx_start]
                        if len(tmp1) != 0:
                            results_static[data_tmp['classId']].extend(tmp1)
                        else:
                            results_static[data_tmp['classId']].append(data_tmp['text'])
                    else:
                        results_static[data_tmp['classId']].append(data_tmp['text'])
            if flag_1 is True:
                sent_with_entity += 1
                sent_we.append(cur_sen)
            sent_all += 1
            sent_z.append(cur_sen)
            f.close()

    #

    sl = []
    biosdata = ['train.char.bios','test.char.bios','dev.char.bios']
    for p1 in biosdata:
        f1 = os.path.join(BIOS_dataset, p1)
        f = open(f1, 'r')
        c = 0
        d = 0
        for kk in f.readlines():
            d += 1
            if kk == '\n':
                sl.append(d)
                d = 0
                c += 1
    #
    train, test, dev = get_train_test_dev_info(BIOS_dataset)
    #
    kz_len = {}
    for i in results_static:
        kz_len[i] = []
        for k in results_static[i]:
            kz_len[i].append(len(k))
    #
    print('# doc\t%d' % (len(os.listdir(json_dataset))))
    print('------------------------')
    print('# sentences\t%d' % (sent_all))
    print('------------------------')
    print('# sentences with ann\t%d' % (sent_with_entity))
    print('------------------------')
    print('Avg sentences len \t%.2f' % (np.average(sl)))
    print('------------------------')

    tn = 0
    print('Type \t\t Num \t\t Train \t\t Dev \t\t Test \t\t Avg len')
    print('------------------------')
    for i in results_static:
        entity_type = ner_label_map[i]
        print('%s \t\t %d \t\t %d \t\t %d \t\t %d\t\t %.2f' % (
            entity_type, len(results_static[i]), train[entity_type], dev[entity_type], test[entity_type],
            np.mean(kz_len[i])))
        tn += len(results_static[i])
    print('------------------------')
    print('Total \t\t %d \t\t %d \t\t %d \t\t %d' % (
        tn, sum([train[i] for i in train]), sum(dev[i] for i in dev), sum(test[i] for i in test)))
    print('\n')
