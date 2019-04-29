#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/27 19:59:00
"""

from pytest import approx
import json
from unittest import TestCase

from projects import PaperClassifierPredictor

from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

class TestAcademicPaperPredictor(TestCase):

    def test_academic_paper_predictor(self):

        # predcitor = PaperClassifierPredictor()

        inputs = {
            "title": "Interferring Discourse Relations in Context",
            "paperAbstract": (
                "We investigate various contextual effects on text "
                "interpretation, and account for them by providing "
                "contextual constraints in a logical theory of text "
                "interpretation. On the basis of the way these constraints "
                "interact with the other knowledge sources, we draw some "
                "general conclusions about the role of domain-specific "
                "information, top-down and bottom-up discourse information "
                "flow, and the usefulness of formalisation in discourse theory."
            )
        }

        archive_file_path = "/Users/panxu/serialize/academic_paper_classify2/model.tar.gz"
        archive = load_archive(archive_file_path)
        predictor = Predictor.from_archive(archive, "paper_classifer")

        result = predictor.predict_json(inputs=inputs)

        print(json.dumps(result))

        label = result["label"]

        assert label in ['AI', 'ML', 'ACL']

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)

