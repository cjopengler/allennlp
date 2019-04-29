#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/27 21:35:00
"""

import json
import unittest
from unittest import TestCase

from pytest import approx
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from projects.predictors import PersonNamePredictor


class TestPersonNamePredictor(TestCase):


    def test_person_name_predictor(self):

        archive_file_path = "~/serialize/person_name_classify/model.tar.gz"

        archive = load_archive(archive_file_path)

        predictor = Predictor.from_archive(archive=archive,
                                           predictor_name="PersonNamePredictor")

        inputs = {"name": "Xi Jinping"}

        result = predictor.predict_json(inputs=inputs)



        print(json.dumps(result))

        self.assertEqual(result["label"], 0)

        class_probabilities = result.get("class_probalities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)

        inputs = {"name": "Jacob Zuma"}

        result = predictor.predict_json(inputs=inputs)

        print(json.dumps(result))

        self.assertEqual(result["label"], 1)
        class_probabilities = result.get("class_probalities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)