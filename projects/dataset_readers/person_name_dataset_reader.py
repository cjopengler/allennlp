#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
人名的data set reader

Authors: panxu(panxu@baidu.com)
Date:    2019/04/22 19:05:00
"""
import json
from typing import Iterable

from allennlp.data import DatasetReader
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.fields import TextField
from allennlp.data.fields import LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance


@DatasetReader.register("PersonNameDatasetReader")
class PersonNameDatasetReader(DatasetReader):

    def __init__(self, tokenizer: Tokenizer, lazy=True):
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:

            for line in f:
                item = json.loads(line.strip())
                name = item["name"]
                label = item.get("label", None)

                yield self.text_to_instance(name=name, label=label)

    def text_to_instance(self, name: str, label: int = None, *inputs) -> Instance:

        name_tokens = self._tokenizer.tokenize(name)
        name_field = TextField(tokens=name_tokens,
                               token_indexers={"characters": SingleIdTokenIndexer(namespace="tokens")})

        fields = {"name": name_field}

        if label is not None:
            label_field = LabelField(label=label,
                                     label_namespace="labels",
                                     skip_indexing=True)

            fields["label"] = label_field

        return Instance(fields)