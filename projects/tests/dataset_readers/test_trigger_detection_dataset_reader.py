#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
测试 trigger detection dataset reader

Authors: panxu(panxu@baidu.com)
Date:    2019/04/29 17:14:00
"""

import logging
import os
from allennlp.data import DatasetReader
from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from projects.dataset_readers import TriggerDetectionDatasetReader



class TestTriggerDetectionDatasetReader(AllenNlpTestCase):

    def setUp(self):
        super().setUp()

        logging.debug("project root {}".format(self.PROJECT_ROOT))
        logging.debug("test dir {}".format(self.TEST_DIR))
        logging.debug("test dir root {}".format(self.TESTS_ROOT))

    def test_trigger_detection_dataset_reader(self):

        params_filepath = os.path.join(self.PROJECT_ROOT,
                                       "data/projects/tests/trigger_detection.jsonnet")

        logging.debug("params_filepath: {}".format(params_filepath))

        params = Params.from_file(params_file=params_filepath)

        # 提取dataset reader params
        params = params["dataset_reader"]
        reader = DatasetReader.from_params(params=params)

        training_data_filepath = os.path.join(self.PROJECT_ROOT,
                                              "data/projects/tests/training.data")

        logging.debug("training data filepath: {}".format(training_data_filepath))

        instances = reader.read(file_path=training_data_filepath)
        instances = ensure_list(instances)

        instance = instances[0]
        sentence = "江西景德镇客车相撞致5人死亡20余人受伤"
        labels = ["O", "O", "O", "O", "O", "O", "O",
                  "B-TC", "I-TC", "O", "O", "O", "O",
                  "O", "O", "O", "O", "O", "O", "O"]

        self.assertEqual(len(instance.fields), 3)


        sentence_field = instance.fields["sentence"]

        labels_field = instance.fields["labels"]
        metadata_field = instance.fields["metadata"]

        self.assertEqual(len(sentence_field.tokens), len(sentence))

        self.assertListEqual([t.text for t in sentence_field.tokens],
                             [_ for _ in sentence])

        self.assertEqual(len(labels_field.labels), len(labels))
        self.assertListEqual(labels_field.labels, labels)

        self.assertEqual(metadata_field.metadata["sentence"], sentence)





