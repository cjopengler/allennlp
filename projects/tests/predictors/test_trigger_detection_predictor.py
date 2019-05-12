#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
测试 trigger detection

Authors: panxu(panxu@baidu.com)
Date:    2019/05/12 12:47:00
"""

import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from projects.predictors import TriggerDetectionPredictor


class TestTriggerDetectionPredictor(AllenNlpTestCase):
    """
    测试 trigger detection predictor
    """

    def test_trigger_detection_predictor(self):
        """
        测试
        :return:
        """

        inputs = {"hit": 1,
                  "text": "京珠高速郑州往焦作方向19辆车连环相撞",
                  "type": "title",
                  "label": ["O", "O", "O", "O", "O", "O", "O", "O",
                            "O", "O", "O", "O", "O", "O", "O", "B-TC",
                            "I-TC", "I-TC", "I-TC"]}

        archive_filepath = "~/serialize/trigger_detection/model.tar.gz"

        archive = load_archive(archive_file=archive_filepath)
        predictor = Predictor.from_archive(archive=archive,
                                           predictor_name="TriggerDetectionPredictor")

        result = predictor.predict_json(inputs)

        print(json.dumps(result))