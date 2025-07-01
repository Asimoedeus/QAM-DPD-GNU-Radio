"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr

import torch
import torch.nn as nn

class GRUBackbone(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc_out(out)
        return out

class CoreModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=15, num_layers=1, backbone_type="deltagru"):
        super().__init__()
        if backbone_type == "deltagru":
            self.backbone = GRUBackbone(input_size, hidden_size, num_layers)
        # 其他类型...

    def forward(self, x):
        return self.backbone(x)
# ---- ↑↑↑ 网络结构定义只需贴一次 ↑↑↑ ----

    
def build_features(x):
    """
    输入：x shape [N,] complex
    输出：features shape [N,6] float32
    """
    I = x.real
    Q = x.imag
    mag = np.abs(x)
    mag2 = mag ** 2
    # 滞后1拍，最前面补0
    I_delay = np.roll(I, 1); I_delay[0]=0
    Q_delay = np.roll(Q, 1); Q_delay[0]=0
    features = np.stack([I, Q, mag, mag2, I_delay, Q_delay], axis=1)
    return features.astype(np.float32)

class blk(gr.sync_block):

    """ DPD Neural Network Block (PyTorch) """
    def __init__(self, model_path="C:/Programming/pythonProjects/QAM-In-GNU-Radio-main/Rapp_PA_save_file/DPD_S_0_M_DELTAGRU_H_15_F_200_P_1067_THX_0.000_THH_0.000.pt"):
        gr.sync_block.__init__(self,
            name='DPD_NN_block',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.model = CoreModel(input_size=6, hidden_size=15, num_layers=1, backbone_type="deltagru") # 按训练时参数设置
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def work(self, input_items, output_items):
        x = input_items[0]
        x_feats = build_features(x)  # shape [N,6]
        x_tensor = torch.from_numpy(x_feats[None, ...]).to(self.device)
        with torch.no_grad():
            y_tensor = self.model(x_tensor)
        y_np = y_tensor.cpu().numpy()[0]
        y_complex = y_np[:,0] + 1j*y_np[:,1]
        output_items[0][:] = y_complex.astype(np.complex64)
        return len(output_items[0])


