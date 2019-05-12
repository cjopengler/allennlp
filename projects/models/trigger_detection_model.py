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
                 crf: bool = True,
                 constrain_crf_decoding: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        """
        初始化
        :param vocab:
        :param sentence_embedder:
        :param encoder:
        :param calculate_span_f1: 是否计算f1 metric
        :param crf: True: 使用crf; False: 不使用crf.
        :param constrain_crf_decoding: 对crf decoding做限制. 只有当crf参数是True的时候起作用
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
        self._crf = None

        if crf:
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

        if self._crf:
            best_path = self._crf.viterbi_tags(logits=logits,
                                               mask=sentence_mask)
            # 从best_path中提取出tag
            # 在best path中返回的是 tag 和 socre 二元组
            predicted_labels = [tag for tag, score in best_path]

            output_dict["viterib_labels"] = predicted_labels

            # class_probilities 使用viterbi得到tag重新计算. 如果命中了viterbi得到tag
            # 就让这个tag的概率为1。如果没有命中，那么就让这个tag的概率为0

            # 先产生一个与logits一样维度的class_probilities
            class_probilities = logits * 0.0

            # 对其中命中了tag的需要设置概率为1.0

            # i 相当于batch index;
            # instance_labels 是一个tag list是1维的，也就是一个instance的tag
            for i, instance_labels in enumerate(predicted_labels):

                # j 相当于每个sequnce token index; tag_id 表示那个tag被viterbi选出来了
                # 这里要注意的是因为tag 是0-label_size 进行编码的，所以可以这样使用.
                for j, tag_id in enumerate(instance_labels):
                    class_probilities[i, j, tag_id] = 1.0

            output_dict["class_probilities"] = class_probilities

        else:
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

            if self._crf:
                # 注意是前面有个负号
                loss = -self._crf(logits, labels, sentence_mask)
            else:
                loss = sequence_cross_entropy_with_logits(logits,
                                                          labels,
                                                          sentence_mask)

            output_dict["loss"] = loss

            # 计算metric

            if self._crf:
                # 在使用了crf之后 对metric的计算就要使用viterbi之后的结果, 也就是class_probilitis
                class_probilities = output_dict["class_probilities"]
                for _, metric in self._metric.items():
                    metric(class_probilities, labels, sentence_mask)

                if self._f1_metric:
                    self._f1_metric(class_probilities,
                                    labels,
                                    sentence_mask)

            else:
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


        viterbi_labels = output_dict.get("viterib_labels", None)

        if viterbi_labels is None:
            class_probilities = output_dict["class_probilities"]

            # 获取argmax值
            indeces = torch.argmax(class_probilities, dim=-1)

            # 转换到label
            indeces_size = indeces.size()

            reshaped_indeces = indeces.view(-1).cpu().numpy()
            labels = [self.vocab.get_token_from_index(i, namespace="labels") for i in reshaped_indeces]
            labels = np.array(labels).reshape(indeces_size)

            output_dict["labels"] = labels
        else:
            labels = [[self.vocab.get_token_from_index(label_id, namespace="labels") for label_id
                       in instance_labels] for instance_labels in viterbi_labels]
            output_dict["labels"] = labels

        return output_dict
