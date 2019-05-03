#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/26 18:45:00
"""

class SplitDataset(object):
    """
    切分数据集成训练集、验证集和测试集
    """

    def split(self,
              input_filepath: str,
              traing_dataset_filepath: str,
              validation_dataset_filepath: str,
              test_dataset_filepath: str = None):
        """
        切分
        :param input_filepath:
        :param traing_dataset_filepath:
        :param validation_dataset_filepath:
        :param test_dataset_filepath:
        :return:
        """
        pass