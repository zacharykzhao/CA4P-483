# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2021/11/29
# System: linux

import os
from tqdm import tqdm
'''

target format:
    印   B-LOC
    度   M-LOC
    河   E-LOC
    流   O
    经   O
    印   B-GPE
    度   E-GPE
'''

if __name__ == '__main__':
    tar_path = "../train_dev_test"

    for zty in os.listdir(tar_path):
        tar_file_name = zty.replace("bi", "bme")
        save_file = os.path.join(tar_path,tar_file_name)
        f = open(tar_path + "/" + zty, 'r', encoding='utf-8')
        data = f.readlines()
        fw = open(save_file, 'w', encoding='utf-8')
        for idx in tqdm(range(0, len(data)-1)):
            tmp = data[idx]
            if tmp == '\n':
                if data[idx+1] == '\n':
                    continue
                else:
                    fw.write('\n')
            else:
                char = tmp.split('\t')[0]
                label = tmp.split('\t')[-1]
                nxt_label = data[idx+1].split('\t')[-1]
                if label.startswith('I') and (not nxt_label.startswith('I')):
                    label = 'E' + label[1:]
                elif label.startswith('S'):
                    label = 'B' + label[1:]
                fw.write(char +'\t' + label)



