#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/20 19:02:00
"""

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from projects.dataset_readers.semantic_scholar_dataset_reader import SemanticScholarDatasetReader

class TestSemanticScholarDatasetReader(AllenNlpTestCase):

    def test_reader(self):

        file_path = "data/test_s2_papers.jsonl"
        reader = SemanticScholarDatasetReader()


        instances = reader.read(file_path=file_path)

        instances = ensure_list(instances)

        self.assertEqual(len(instances), 10)

        expect_instance0 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                     "abstract": ["We", "investigate", "various", "contextual", "effects"],
                     "venue": "ACL"}

        instance0 = instances[0]

        title_tokens = instance0.fields["title"].tokens
        self.assertListEqual([t.text for t in title_tokens], expect_instance0["title"])

        abstract_tokens = instance0.fields["abstract"].tokens[0:5]
        self.assertListEqual([t.text for t in abstract_tokens], expect_instance0["abstract"])

        label = instance0.fields["label"].label
        self.assertEqual(label, expect_instance0["venue"])





