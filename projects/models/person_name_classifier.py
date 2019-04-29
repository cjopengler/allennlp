#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
名字分类器

Authors: panxu(panxu@baidu.com)
Date:    2019/04/23 13:48:00
"""
from typing import Dict, List, Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2VecEncoder
from allennlp.modules import FeedForward

from allennlp.data import Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.fields import LabelField
from allennlp.nn import RegularizerApplicator
from allennlp.nn import InitializerApplicator
from allennlp.nn import util

from allennlp.training.metrics import CategoricalAccuracy


@Model.register("PersonNameClassifier")
class PersonNameClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 feed_forward: FeedForward,
                 regularizer: RegularizerApplicator = None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab=vocab, regularizer=regularizer)

        self._text_embedder = text_embedder
        self._encoder = encoder
        self._feed_forward = feed_forward

        # 定义loss

        self._loss = CrossEntropyLoss()
        self._metric = {"accuracy": CategoricalAccuracy()}

        # 初始化所有
        initializer(self)

    def forward(self,
                name: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        name_embedding = self._text_embedder(name)
        name_mask = util.get_text_field_mask(name)

        encoding = self._encoder(name_embedding, name_mask)

        logits = self._feed_forward(encoding)

        output_dict = {"logits": logits}

        if label is not None:
            loss = self._loss(logits, label)
            output_dict["loss"] = loss

            for metric in self._metric.values():
                metric(logits, label)

        if metadata is not None:
            output_dict["name"] = [x["name"] for x in metadata]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = {}
        for metric_name, metric in self._metric.items():
            result[metric_name] = metric.get_metric(reset)

        return result

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        decode 产生label
        :param output_dict:
        :return:
        """

        logits = output_dict["logits"]

        class_probalities = F.softmax(logits)

        class_probalities = class_probalities.cpu().numpy()

        output_dict["class_probalities"] = class_probalities

        argmax_indecies = np.argmax(class_probalities, axis=-1)

        labels = [_ for _ in argmax_indecies]

        output_dict["label"] = labels

        return output_dict








