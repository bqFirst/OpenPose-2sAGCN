#! /user/bin/env python3
# coding=utf-8
# @Time   : 2020/3/14 0014 18:14
# @Author : wangw
# @File   : test.py
# @Desc   :

import yaml
import json
import numpy as np

file = './data/kinetics_raw/kinetics_val_label.json'

val_label = json.load(open(file, 'rb'))
labels = {}
for i, x in enumerate(val_label):
    labels[val_label[x]['label']] = val_label[x]['label_index']

json.dump(labels, open('labels.json', 'w'))
data = json.load(open('labels.json', 'rb'))
new_data = {v: k for k, v in data.items()}
print(new_data[172])