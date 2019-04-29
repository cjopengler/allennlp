#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2019/04/24 09:18:00
"""

"""Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()

        print("original size", original_size)

        input = input.view(-1, input.size(self.dim))

        print("input", input)

        dim = 1
        number_of_logits = input.size(dim)

        print("number of logits", number_of_logits)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        print("input", input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)

        print("zs1", zs)
        zs = zs[0]

        print("zs2", zs)

        range = torch.range(start=1, end=number_of_logits, device=device).view(1, -1)

        print("range1", range)
        range = range.expand_as(zs)
        print("range2", range)

        # Determine sparsity of projection
        bound = 1 + range * zs

        print("bound", bound)

        cumulative_sum_zs = torch.cumsum(zs, dim)

        print("cumulative_sum_zs", cumulative_sum_zs)

        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())

        print("is_gt", is_gt)

        k = torch.max(is_gt * range, dim, keepdim=True)

        print("k1", k)
        k = k[0]

        print("k2", k)
        # Compute threshold function
        zs_sparse = is_gt * zs

        print("zs sparse", zs_sparse)

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k

        print("taus1", taus)

        taus = taus.expand_as(input)

        print("taus2", taus)


        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        print("output", self.output)

        output = self.output.view(original_size)

        print("output2", output)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


if __name__ == '__main__':

    sparsemax = Sparsemax(dim=1)
    softmax = torch.nn.Softmax(dim=1)

    logits = torch.randn(2, 5)

    # print(logits.dim)
    # print(logits.size())
    # logits = torch.tensor([[0.5, 0.2, 0.1], [0.01, 0.09, 0.8]])
    # logits = torch.tensor([[1., 2., 3.], [6., 4., 5.]])
    print("dim", logits.dim)
    print("\nLogits")
    print(logits)

    softmax_probs = softmax(logits)
    print("\nSoftmax probabilities")
    print(softmax_probs)

    sparsemax_probs = sparsemax(logits)
    print("\nSparsemax probabilities")
    print(sparsemax_probs)
    sparsemax.backward(sparsemax_probs)
