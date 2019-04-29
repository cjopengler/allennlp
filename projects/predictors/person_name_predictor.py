#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
person name 分类器的预测器

Authors: panxu(panxu@baidu.com)
Date:    2019/04/27 21:35:00
"""

from allennlp.predictors import Predictor
from common import JsonDict
from data import Instance


@Predictor.register("PersonNamePredictor")
class PersonNamePredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        name = json_dict["name"]

        instance = self._dataset_reader.text_to_instance(name=name)

        return instance