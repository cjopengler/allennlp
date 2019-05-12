#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
trigger detection predictor

Authors: panxu(panxu@baidu.com)
Date:    2019/05/12 11:39:00
"""

from allennlp.predictors import Predictor
from common import JsonDict
from data import Instance


@Predictor.register("TriggerDetectionPredictor")
class TriggerDetectionPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["text"]

        return self._dataset_reader.text_to_instance(sentence=sentence)