#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
触发词识别 dataset reader

Authors: panxu(panxu@baidu.com)
Date:    2019/04/29 14:41:00
"""

import json
from typing import Iterable, List

from allennlp.data import DatasetReader
from allennlp.data import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.fields import MetadataField


@DatasetReader.register("TriggerDetectionDatasetReader")
class TriggerDetectionDatasetReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexer: TokenIndexer,
                 lazy: bool = False):
        """
        初始化 trigger detection dataset reader
        :param tokenizer: tokenizer
        :param token_indexer: token indexer
        :param lazy:
        """
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer
        self._token_indexer = token_indexer


    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                item = json.loads(line)

                sentence = item["text"]
                labels = item.get("label", None)


                yield self.text_to_instance(sentence=sentence,
                                            labels=labels)

    def text_to_instance(self, sentence: str, labels: List[str] = None) -> Instance:

        tokens = [_ for _ in self._tokenizer.tokenize(sentence)]

        sentence_field = TextField(tokens=tokens,
                                   token_indexers={"character": self._token_indexer})

        fields = {"sentence": sentence_field}

        if labels:
            label_field = SequenceLabelField(labels=labels,
                                             sequence_field=sentence_field,
                                             label_namespace="labels")

            fields["labels"] = label_field

        fields["metadata"] = MetadataField(metadata={"sentence": sentence})

        return Instance(fields=fields)