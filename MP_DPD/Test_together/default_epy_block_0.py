#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# Embedded Python Block: Memory Polynomial DPD for GNU Radio
# Implements a digital predistorter using a memory polynomial model

import numpy as np
from gnuradio import gr

class MP_DPD(gr.sync_block):
    """
    Memory Polynomial Digital Predistortion (DPD) Block.

    y[n] = sum_{m=0}^Q sum_{p=1,3,...,K} a_{p,m} * x[n-m] * |x[n-m]|^(p-1)

    Args:
        coeffs (list of lists): complex coefficients a_{p,m} of shape (P, Q+1),
            where P = (K+1)/2 rows for orders p=1,3,...,K and Q+1 columns for delays m=0..Q.
        K (int): maximum odd nonlinear order (e.g., 5).
        Q (int): memory depth (e.g., 9 delays).
    """
    def __init__(self, coeffs=None, K=5, Q=1):
        gr.sync_block.__init__(
            self,
            name='MP_DPD',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        # 参数校验
        if coeffs is None:
           # raise ValueError("MP_DPD: must supply 'coeffs' as a list of lists.")
           coeffs = [
                [9.138884e-01+3.676318e-02j, 4.883846e-02-9.244402e-02j],
                [3.472112e-02+1.207297e-01j, -4.867465e-02+3.718652e-02j],
                [-8.472962e-03-5.038591e-02j, 1.019886e-02+3.677094e-02j],
            ]
        self.coeffs = np.array(coeffs, dtype=np.complex64)
        if self.coeffs.ndim != 2:
            raise ValueError("MP_DPD: 'coeffs' must be 2D array-like (orders x delays).")

        # 模型阶数与记忆深度
        self.K = K
        self.Q = Q
        self.orders = list(range(1, K+1, 2))  # [1,3,5,...,K]
        if self.coeffs.shape != (len(self.orders), Q+1):
            raise ValueError(
                f"MP_DPD: coeffs shape must be ({len(self.orders)}, {Q+1}), got {self.coeffs.shape}"
            )

        # 历史样本缓冲区 (保存最近 Q 个样本)
        self.history = np.zeros(self.Q, dtype=np.complex64)

    def work(self, input_items, output_items):
        x_new = input_items[0]
        y_out = output_items[0]
        N = len(x_new)

        # 前端拼接历史样本
        x_full = np.concatenate((self.history, x_new))  # 长度 Q + N

        # 计算 DPD 输出
        y_out[:] = 0
        for idx, p in enumerate(self.orders):
            for m in range(self.Q + 1):
                a_pm = self.coeffs[idx, m]
                # 延迟 m 的输入向量
                x_del = x_full[self.Q - m : self.Q - m + N]
                # 多项式叠加
                y_out += a_pm * x_del * (np.abs(x_del) ** (p - 1))

        # 更新历史样本用于下一个块
        if self.Q > 0:
            self.history = x_full[-self.Q:]

        return len(y_out)
