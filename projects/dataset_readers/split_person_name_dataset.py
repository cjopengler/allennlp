#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
将person name分割成训练集和验证集

Authors: panxu(panxu@baidu.com)
Date:    2019/04/26 19:11:00
"""
import json
from sklearn.model_selection import train_test_split

class SplitPersonNameDataset(object):
    """
    切分person name dataset
    """

    def __call__(self,
                 input_filepath: str,
                 train_filepath: str,
                 val_filepath: str,
                 val_size: float,
                 *args, **kwargs):

        x = list()
        y = list()
        with open(input_filepath) as f:

            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                info = json.loads(line)

                label = info["label"]
                name = info["name"]

                x.append(name)
                y.append(label)


        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=val_size, stratify=y)

        with open(train_filepath, 'w') as f:

            for name, label in zip(x_train, y_train):
                f.write(json.dumps({
                    "name": name,
                    "label": label
                }))
                f.write("\n")

        with open(val_filepath, 'w') as f:

            for name, label in zip(x_test, y_test):
                f.write(json.dumps({
                    "name": name,
                    "label": label
                }))
                f.write("\n")

if __name__ == '__main__':

    input_filepath = "projects/data/person_name_classify/person_name.data"
    train_filepath = "projects/data/person_name_classify/person_name.train.data"
    val_filepath = "projects/data/person_name_classify/person_name.val.data"
    SplitPersonNameDataset()(input_filepath=input_filepath,
                             train_filepath=train_filepath,
                             val_filepath=val_filepath,
                             val_size=0.3)
