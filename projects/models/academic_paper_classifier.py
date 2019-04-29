#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
academic paper classifer

Authors: panxu(panxu@baidu.com)
Date:    2019/04/21 07:37:00
"""
from typing import Dict

import torch

import logging
import os
from typing import Dict, Union, List, Set

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator

from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2VecEncoder
from allennlp.modules import FeedForward
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("paper_classifer")
class AcademicPaperClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedfowrd: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedfowrd = classifier_feedfowrd


        # 初始化所有的变量. 其实每一个torch.module都会自动的初始化。
        # 这里的初始化是 对一些特定的模型进行初始化
        # 所以一般没有必要设置变量
        initializer(self)

        # 定义loss
        self.loss = CrossEntropyLoss()

        # 定义metric
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }

    def forward(self,
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                *inputs) -> Dict[str, torch.Tensor]:
        """
        forword 调用如何传入参数的?
        :param title:
        :param abstract:
        :param label:
        :param inputs:
        :return:
        """

        title_embedder = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)

        encode_title = self.title_encoder(title_embedder, title_mask)

        abstract_embedder = self.text_field_embedder(abstract)
        abstract_mask = util.get_text_field_mask(abstract)

        encode_abstract = self.abstract_encoder(abstract_embedder,
                                                abstract_mask)

        encode = torch.cat([encode_title, encode_abstract], dim=-1)

        logits = self.classifier_feedfowrd(encode)

        output_dict = {'logits': logits}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        输入就是forword 产生额结果
        :param output_dict:
        :return:
        """
        class_probabilities = F.softmax(output_dict["logits"])

        output_dict["class_probabilities"] = class_probabilities

        predictions = class_probabilities.cpu().numpy()

        argmax_indecies = np.argmax(predictions, axis=-1)

        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indecies]

        output_dict["label"] = labels
        return output_dict
