"""
Embedded Python block for ARVTDNN DPD model inference in GNU Radio
"""
import numpy as np
import torch
import torch.nn as nn
from gnuradio import gr

class ARVTDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(ARVTDNN, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 2))  # 输出 I/Q
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_features(x, memory=3, order=3, order_memory=2):
    """
    构建 ARVTDNN 特征：延迟 I/Q + 幅度多项式
    """
    N = len(x) - memory
    feat_dim = (memory + 1) * 2 + (order - 1) * (order_memory + 1)
    feats = np.zeros((N, feat_dim), dtype=np.float32)
    for n in range(N):
        idx = 0
        # 时滞 I/Q
        for m in range(memory + 1):
            feats[n, idx] = x[n + m].real; idx += 1
            feats[n, idx] = x[n + m].imag; idx += 1
        # 幅度多项式
        for m in range(order_memory + 1):
            amp = np.abs(x[n + m])
            for k in range(2, order + 1):
                feats[n, idx] = amp**k; idx += 1
    return feats

class blk(gr.sync_block):
    """ARVTDNN DPD Model Block"""
    def __init__(self, model_path="C:\Programming\pythonProjects\QAM-In-GNU-Radio-main\ILA\dpd_model.pth", memory=3, order=3, order_memory=2, scale=1.0):
        gr.sync_block.__init__(self,
            name="ARVTDNN_DPD",
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self.model_path = model_path
        self.memory = memory
        self.order = order
        self.order_memory = order_memory
        self.scale = scale
        self.device = torch.device('cpu')

        # 加载模型
        input_dim = (self.memory + 1) * 2 + (self.order - 1) * (self.order_memory + 1)
        self.net = ARVTDNN(input_dim).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()

        # 保留历史样本
        self.buf = np.zeros(self.memory, dtype=np.complex64)

    def work(self, input_items, output_items):
        x_in = input_items[0]
        N = len(x_in)
        # 拼接历史与当前
        x = np.concatenate((self.buf, x_in)).astype(np.complex64)
        # 归一化
        x_norm = x / self.scale
        # 特征提取
        feats = extract_features(x_norm, self.memory, self.order, self.order_memory)
        # 推断
        with torch.no_grad():
            tensor = torch.from_numpy(feats).to(self.device)
            pred = self.net(tensor).cpu().numpy()
        # 重构复信号
        dpd = (pred[:, 0] + 1j * pred[:, 1]) * self.scale
        y = dpd.astype(np.complex64)
        # 输出并更新历史
        output_items[0][:] = y
        self.buf = x_in[-self.memory:]
        return N
