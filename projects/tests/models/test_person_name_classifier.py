#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/23 15:45:00
"""
import unittest
from projects.dataset_readers.person_name_dataset_reader import PersonNameDatasetReader
from projects.models.person_name_classifier import PersonNameClassifier
from allennlp.common.testing import ModelTestCase
from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.data.dataset import Batch

class TestPersonNameClassifier(ModelTestCase):

    def setUp(self):
        super().setUp()

        param_file = "data/person_name_classify.jsonnet"
        dataset_file = "data/person_name.traing.data"
        self.set_up_model(param_file=param_file, dataset_file=dataset_file)

    def test_model_train_and_save(self):
        # param_file = "data/person_name_classify.jsonnet"
        self.ensure_model_can_train_save_and_load(param_file=self.param_file)
        print("\n")
        print("finished ---------------")

if __name__ == '__main__':

    unittest.main()