#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def estimate_mp_dpd_coeffs(z, y, K=5, Q=9):
    """
    离线估计 Memory-Polynomial DPD 系数

    参数
    ----
    z : np.ndarray of complex, shape (N,)
        原始发送基带信号 (I + jQ)
    y : np.ndarray of complex, shape (N,)
        对应的 PA 输出信号 (I + jQ)
    K : int
        多项式最高阶（奇数），例如 5
    Q : int
        记忆深度（延迟 0…Q）

    返回
    ----
    coeffs : np.ndarray of complex, shape ((K+1)//2, Q+1)
        DPD 系数矩阵，行对应 p=1,3,…,K，列对应延迟 m=0…Q
    a_vec   : np.ndarray of complex, shape (((K+1)//2)*(Q+1),)
        拉平成一维的系数向量
    """
    z = np.asarray(z).ravel().astype(np.complex64)
    y = np.asarray(y).ravel().astype(np.complex64)
    N = len(z)
    if len(y) != N:
        raise ValueError("z 和 y 长度必须一致")

    # 归一化 PA 输出，使其平均幅度与 z 相同
    gain = np.mean(np.abs(y)) / np.mean(np.abs(z))
    y_norm = y / gain

    P = (K + 1) // 2
    U = np.zeros((N, P * (Q + 1)), dtype=np.complex64)

    # 构造回归矩阵 U
    for q in range(Q + 1):
        y_shift = np.roll(y_norm, q)             # 延迟 q
        for i, p in enumerate(range(1, K+1, 2)):  # p = 1,3,5...
            U[:, q*P + i] = y_shift * (np.abs(y_shift) ** (p - 1))

    # 最小二乘求解： U @ a ≈ z
    a_vec, *_ = np.linalg.lstsq(U, z, rcond=None)
    coeffs = a_vec.reshape(P, Q + 1)

    return coeffs, a_vec

if __name__ == "__main__":
    # ————————————
    # 1) 读取 CSV
    #    假设 train_input.csv, train_output.csv 放在脚本同目录
    df_in  = pd.read_csv("../GMP_PA/datasets/train_input.csv")   # 列名 I, Q
    df_out = pd.read_csv("../GMP_PA/datasets/train_output.csv")  # 列名 I, Q

    z = df_in["I"].values + 1j * df_in["Q"].values
    y = df_out["I"].values + 1j * df_out["Q"].values

    # 2) 估计系数
    K = 5   # Max odd order
    Q = 1   # Memory depth
    coeffs_matrix, a_vec = estimate_mp_dpd_coeffs(z, y, K=K, Q=Q)

    # 3) 打印成 Python 列表格式，拷贝到 GNU Radio Block
    P = (K + 1)//2
    print("coeffs = [")
    for row in coeffs_matrix:
        entries = ", ".join(f"{c.real:.6e}{c.imag:+.6e}j" for c in row)
        print(f"    [{entries}],")
    print("]")
    '''
    coeffs = [
    [9.138884e-01+3.676318e-02j, 4.883846e-02-9.244402e-02j],
    [3.472112e-02+1.207297e-01j, -4.867465e-02+3.718652e-02j],
    [-8.472962e-03-5.038591e-02j, 1.019886e-02+3.677094e-02j],
    ]
    '''
    # (可选) 保存为 .npy 或 .npz
    # np.save("dpd_coeffs_matrix.npy", coeffs_matrix)
    # np.savez("dpd_coeffs.npz", matrix=coeffs_matrix, vector=a_vec)

    # 4) 你就可以在 GRC 中这样实例化：
    #    from your_block_module import MP_DPD
    #    block = MP_DPD(coeffs=coeffs, K=5, Q=9)
