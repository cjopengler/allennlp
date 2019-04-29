#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
paper 分类器的predictor

Authors: panxu(panxu@baidu.com)
Date:    2019/04/27 19:27:00
"""

from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.data import Instance


@Predictor.register("paper_classifer")
class PaperClassifierPredictor(Predictor):
    """
    paper 分类predictor
    """

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict["title"]
        abstract = json_dict['paperAbstract']

        instance = self._dataset_reader.text_to_instance(title=title,
                                                         abstract=abstract)
        return instance





