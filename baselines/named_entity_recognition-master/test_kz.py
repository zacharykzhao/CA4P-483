# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2022/5/31
# System: linux

'''
This for test BiLSTM-CRF
'''
import os
from data import build_corpus, build_map
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate


def get_word_list(file_name, make_vocab=False):
    f = open(file_name, 'r')
    l = f.readline()
    word_list = []
    tag_list = []
    word_lists = []
    tag_lists = []
    while l:
        if l != '\n':
            tmp = l.replace('\n','').split('\t')
            word_list.append(tmp[0])
            tag_list.append(tmp[-1])
        else:
            if word_list == []:
                word_list = []
                tag_list = []
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
        l = f.readline()

    f.close()
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists



if __name__ == '__main__':
    data_root = '/home/zachary/Documents/Code/NER/CRFPP_test/data/'
    train_file = os.path.join(data_root,
                              'BME_character_level_train_all_0531.data')
    dev_file = os.path.join(data_root,
                            'BME_character_level_dev_all_0531.data')
    test_file = os.path.join(data_root,
                             'BME_character_level_test_all_0531.data')
    train_word_lists, train_tag_lists, word2id, tag2id = get_word_list(train_file,make_vocab=True)
    dev_word_lists, dev_tag_lists = get_word_list(dev_file)
    test_word_lists, test_tag_lists = get_word_list(test_file)
    #

    # 训练评估CRF模型
    # print("正在训练评估CRF模型...")
    # crf_pred = crf_train_eval(
    #     (train_word_lists, train_tag_lists),
    #     (test_word_lists, test_tag_lists)
    # )

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )

    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )

    # 训练评估BI-LSTM模型
    print("正在训练评估双向LSTM模型...")
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        crf=False
    )

    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id
    )

    ensemble_evaluate(
        [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
        test_tag_lists
    )

