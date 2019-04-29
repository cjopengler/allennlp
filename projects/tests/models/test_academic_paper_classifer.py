#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/22 09:03:00
"""

import logging
import unittest
from projects.dataset_readers.semantic_scholar_dataset_reader import SemanticScholarDatasetReader
from projects.models.academic_paper_classifier import AcademicPaperClassifier

from allennlp.common.testing import ModelTestCase
from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.data.dataset import Batch


class TestAcademicPaperClassifer(ModelTestCase):

    def setUp(self):
        super().setUp()
        param_file = "data/academic_paper_classifier.json"
        dataset_file = "data/test_s2_papers.jsonl"

        super().set_up_model(param_file=param_file,
                             dataset_file=dataset_file)

    def test_model(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


if __name__ == '__main__':
    unittest.main()