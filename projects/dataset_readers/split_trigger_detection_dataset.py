#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
将训练数据拆分成 traing 和 validate

Authors: panxu(panxu@baidu.com)
Date:    2019/05/02 07:55:00
"""

import json
from projects import SplitDataset

from sklearn.model_selection import train_test_split

class SplitTriggerDetectionDataset(SplitDataset):

    def __call__(self,
                 input_filepath: str,
                 traing_dataset_filepath: str,
                 validation_dataset_filepath: str,
                 test_dataset_filepath: str = None):

        self.split(input_filepath=input_filepath,
                   traing_dataset_filepath=traing_dataset_filepath,
                   validation_dataset_filepath=validation_dataset_filepath,
                   test_dataset_filepath=test_dataset_filepath)

    def split(self,
              input_filepath: str,
              traing_dataset_filepath: str,
              validation_dataset_filepath: str,
              test_dataset_filepath: str = None):

        # 先hit和非hit进行spilit

        hits = list()
        items = list()
        with open(input_filepath) as f:
            for line in f:
                line = line.strip()

                item = json.loads(line)

                hits.append(item["hit"])
                items.append(item)


        x_train, x_test, y_train, y_test = train_test_split(items,
                                                            hits,
                                                            test_size=0.2,
                                                            stratify=hits)

        with open(traing_dataset_filepath, 'w') as train_f:

            for x, y in zip(x_train, y_train):
                train_f.write(json.dumps(x, ensure_ascii=False))
                train_f.write("\n")

        with open(validation_dataset_filepath, 'w') as val_f:

            for x, y in zip(x_test, y_test):
                val_f.write(json.dumps(x, ensure_ascii=False))
                val_f.write("\n")

if __name__ == '__main__':
    input_filepath = "projects/data/event/dataset.data"
    train_filepath = "projects/data/event/train_dataset.data"
    val_filepath = "projects/data/event/val_dataset.data"
    SplitTriggerDetectionDataset()(input_filepath=input_filepath,
                                   traing_dataset_filepath=train_filepath,
                                   validation_dataset_filepath=val_filepath)