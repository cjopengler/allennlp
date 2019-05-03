#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
触发词识别模型

Authors: panxu(panxu@baidu.com)
Date:    2019/04/29 18:23:00
"""
from typing import Dict, List, Any

import numpy as np
import torch

from torch.nn import Linear
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models import Model


from allennlp.nn import RegularizerApplicator
from allennlp.nn import InitializerApplicator

from allennlp.modules import TextFieldEmbedder
from allennlp.modules import TimeDistributed
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.training.metrics import Metric

from allennlp.common import Params


@Model.register("TriggerDetectionModel")
class TriggerDetectionModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_encoding: str,
                 label_namespace: str,
                 constrain_crf_decoding: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        """
        初始化
        :param vocab:
        :param sentence_embedder:
        :param encoder:
        :param calculate_span_f1: 是否计算f1 metric
        :param constrain_crf_decoding: 对crf decoding做限制
        :param initializer:
        :param regularizer:
        """

        super().__init__(vocab=vocab,
                         regularizer=regularizer)

        self._sentence_embedder = sentence_embedder
        self._encoder = encoder

        self._num_class = vocab.get_vocab_size("labels")
        self._tag_project_layer = TimeDistributed(Linear(in_features=self._encoder.get_output_dim(),
                                                         out_features=self._num_class))

        self._metric = {"accuracy": CategoricalAccuracy()}

        self._f1_metric = SpanBasedF1Measure(vocabulary=vocab,
                                             tag_namespace=label_namespace,
                                             label_encoding=label_encoding)

        # 增加crf

        constraints = None
        if constrain_crf_decoding:
            constraints = allowed_transitions(label_encoding,
                                              vocab.get_index_to_token_vocabulary(label_namespace))

        self._crf = ConditionalRandomField(num_tags=self._num_class,
                                           constraints=constraints,
                                           include_start_end_transitions=True)

        initializer(self)

    # @classmethod
    # def from_params(cls, vocab: Vocabulary, params: Params, **extras) -> "TriggerDetectionModel":
    #
    #     sentence_embedder = TextFieldEmbedder.from_params(params.pop("sentence_embedder"),
    #                                                       vocab=vocab)
    #     encoder = Seq2SeqEncoder.from_params(params.pop("encoder"),
    #                                          vocab=vocab)
    #     f1_measure = Metric.from_params(params.pop("f1_measure"),
    #                                     vocabulary=vocab)
    #
    #     return cls(vocab=vocab,
    #                sentence_embedder=sentence_embedder,
    #                encoder=encoder,
    #                f1_measure=f1_measure)


    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: Dict[str, torch.Tensor] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        output_dict = dict()

        sentence_embedding = self._sentence_embedder(sentence)
        sentence_mask = get_text_field_mask(sentence)

        # 通过embedding来获取batch_size, squence_length
        batch_size, sequence_length, _ = sentence_embedding.size()

        sentence_encoding = self._encoder(sentence_embedding,
                                          sentence_mask)

        logits = self._tag_project_layer(sentence_encoding)

        output_dict["logits"] = logits

        # 计算每个token上的label 概率
        # 变成2维
        reshape_logits = logits.view(-1, self._num_class)
        # 全部执行softmax
        class_probilities = F.softmax(reshape_logits, dim=-1)

        # 还原到原始的size()
        class_probilities = class_probilities.view(batch_size, sequence_length, -1)

        output_dict["class_probilities"] = class_probilities

        if labels is not None:
            # 计算loss
            loss = sequence_cross_entropy_with_logits(logits,
                                                      labels,
                                                      sentence_mask)

            output_dict["loss"] = loss

            # 计算metric
            for _, metric in self._metric.items():
                metric(logits, labels, sentence_mask)

            if self._f1_metric:
                self._f1_metric(logits,
                                labels,
                                sentence_mask)

        if metadata is not None:
            output_dict["sentence"] = [x["sentence"] for x in metadata]

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metric_result = dict()

        for metric_name, metric in self._metric.items():
            metric_result[metric_name] = metric.get_metric(reset)

        if self._f1_metric:
            metric_result.update(self._f1_metric.get_metric(reset))

        return metric_result

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        class_probilities = output_dict["class_probilities"]

        # 获取argmax值
        indeces = torch.argmax(class_probilities, dim=-1)

        # 转换到label
        indeces_size = indeces.size()

        reshaped_indeces = indeces.view(-1).cpu().numpy()
        labels = [self.vocab.get_token_from_index(i, namespace="labels") for i in reshaped_indeces]
        labels = np.array(labels).reshape(indeces_size)

        output_dict["labels"] = labels

        return output_dict









