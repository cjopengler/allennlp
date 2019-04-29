#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/23 11:31:00
"""

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.params import Params
from allennlp.data import Vocabulary

from projects.dataset_readers.person_name_dataset_reader import PersonNameDatasetReader


class TestPersonNameDatasetReader(AllenNlpTestCase):

    def test_reader(self):
        dataset_file_path = "data/person_name.traing.data"

        reader = PersonNameDatasetReader(tokenizer=CharacterTokenizer(lowercase_characters=True))

        instances = ensure_list(reader.read(dataset_file_path))

        self.assertEqual(len(instances), 2)

        instance0 = instances[0]

        tokens = instance0.fields["name"].tokens

        self.assertListEqual([t.text for t in tokens],
                             ['x', 'i', ' ', 'j', 'i', 'n', 'p', 'i', 'n', 'g'])

        print("label", instance0.fields["label"].label)

        vocab = Vocabulary.from_instances(instances=instances)
        labels = vocab.get_index_to_token_vocabulary("labels")
        print("labels", labels)
        labels = vocab.get_token_to_index_vocabulary("labels")
        print("token and index", labels)


    def test_reader_from_param_file(self):
        params_file_path = "data/person_name_classify.jsonnet"
        params = Params.from_file(params_file_path)

        reader = DatasetReader.from_params(params["dataset_reader"])

        dataset_file_path = "data/person_name.traing.data"

        instances = ensure_list(reader.read(dataset_file_path))

        self.assertEqual(len(instances), 2)

        instance0 = instances[0]

        tokens = instance0.fields["name"].tokens

        self.assertListEqual([t.text for t in tokens],
                             ['x', 'i', ' ', 'j', 'i', 'n', 'p', 'i', 'n', 'g'])