# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2022/6/17
# System: linux


import os
import pickle
if __name__ == '__main__':
    for i in os.listdir("ckpts"):
        print(i)
        f = os.path.join("ckpts", i)
        kf = open(f,'rb')
        data = pickle.load(kf)
        print(data)