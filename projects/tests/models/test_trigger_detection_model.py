#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
测试 trigger detection model

Authors: panxu(panxu@baidu.com)
Date:    2019/04/30 11:35:00
"""
import os
from allennlp.common.testing import ModelTestCase

from projects.dataset_readers import TriggerDetectionDatasetReader
from projects.models import TriggerDetectionModel


class TestTriggerDetectionModel(ModelTestCase):

    def setUp(self):
        super().setUp()

        self.param_filepath = os.path.join(self.PROJECT_ROOT,
                                            "data/projects/tests/trigger_detection.jsonnet")
        self.dataset_filepath = os.path.join(self.PROJECT_ROOT,
                                              "data/projects/tests/training.data")

        self.set_up_model(param_file=self.param_filepath,
                          dataset_file=self.dataset_filepath)

    def test_trigger_dection_model(self):
        self.ensure_model_can_train_save_and_load(self.param_filepath)
