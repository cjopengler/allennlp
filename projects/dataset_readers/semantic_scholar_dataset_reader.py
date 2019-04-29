#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
sematic scholar dataset reader

Authors: panxu(panxu@baidu.com)
Date:    2019/04/20 18:50:00
"""
import json
from typing import Iterable

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance


@DatasetReader.register("ssdr")
class SemanticScholarDatasetReader(DatasetReader):

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f:
                info = json.loads(line.strip())

                title = info["title"]
                abstract = info["paperAbstract"]
                venue = info.get("venue", None) # 是最终的label

                yield self.text_to_instance(title=title,
                                            abstract=abstract,
                                            venue=venue)

    def text_to_instance(self, title: str, abstract: str, venue: str = None, *inputs) -> Instance:
        word_tokenizer = WordTokenizer()
        title_word_tokens = word_tokenizer.tokenize(title)

        title_fieled = TextField(tokens=title_word_tokens,
                                token_indexers={"tokens": SingleIdTokenIndexer()})
        
        abstract_word_tokens = word_tokenizer.tokenize(abstract)
        
        abstract_field = TextField(tokens=abstract_word_tokens, 
                                   token_indexers={"tokens": SingleIdTokenIndexer()})

        fields = {"title": title_fieled, "abstract": abstract_field}

        if venue:
            venue_field = LabelField(label=venue, label_namespace="labels", skip_indexing=False)

            fields["label"] = venue_field

        return Instance(fields=fields)