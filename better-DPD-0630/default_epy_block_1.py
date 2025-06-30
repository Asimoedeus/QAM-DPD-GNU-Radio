"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
import torch
from gnuradio import gr
import torch.nn as nn

class DGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, bidirectional=False, batch_first=True,
                 bias=True):
        super(DGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 6
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.fc_out = nn.Linear(in_features=hidden_size + self.input_size,
                                out_features=self.output_size,
                                bias=self.bias)
        self.fc_hid = nn.Linear(in_features=hidden_size,
                                out_features=hidden_size,
                                bias=self.bias)

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'weight_ih_l0' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])

        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.fc_hid.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, h_0):
        # Feature Extraction
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        cos = i_x / amp
        sin = q_x / amp
        x = torch.cat((i_x, q_x, amp, amp3, sin, cos), dim=-1)
        # Regressor
        out, _ = self.rnn(x, h_0)
        out = torch.relu(self.fc_hid(out))
        out = torch.cat((out, x), dim=-1)
        out = self.fc_out(out)
        return out
        
class CoreModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, backbone_type, window_size=None, num_dvr_units=None, thx=0, thh=0):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PA outputs: I & Q
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.thx = thx
        self.thh = thh
        self.window_size = window_size
        self.num_dvr_units = num_dvr_units
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True

        if backbone_type == 'dgru':
           # from backbones.dgru import DGRU
           self.backbone = DGRU(hidden_size=self.hidden_size,
                                output_size=self.output_size,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional,
                                batch_first=self.batch_first,
                                bias=self.bias)
        else:
            raise ValueError(f"The backbone type '{self.backbone_type}' is not supported. Please add your own "
                             f"backbone under ./backbones and update models.py accordingly.")

        # Initialize backbone parameters
        try:
            self.backbone.reset_parameters()
            print("Backbone Initialized...")
        except AttributeError:
            pass

    def forward(self, x, h_0=None):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None:  # Create initial hidden states if necessary
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Forward Propagate through the RNN
        out = self.backbone(x, h_0)

        return out

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""
    """Apply DPD model to input complex samples"""
    def __init__(self, model_path='C:\Programming\pythonProjects\QAM-In-GNU-Radio-main\Rapp_PA_GNU-Radio\DPD_S_0_M_DGRU_H_15_F_200_P_1319.pt'):
        gr.sync_block.__init__(
            self,
            name='DPD_model',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        model_path = 'C:\Programming\pythonProjects\QAM-In-GNU-Radio-main\Rapp_PA_GNU-Radio\DPD_S_0_M_DGRU_H_15_F_200_P_1319.pt'
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate network with known architecture
        self.net = CoreModel(
            input_size=2,
            hidden_size=15,
            num_layers=1,
            backbone_type='dgru'
        )
        # Load pretrained parameters
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net.to(self.device)

    def work(self, input_items, output_items):
        x = input_items[0]
        # Convert complex data to (N, 2) real-imag tensor
        x_np = np.vstack((x.real, x.imag)).T.astype(np.float32)
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.net(x_tensor).cpu().numpy()[0]
        output_items[0][:] = y[:, 0] + 1j * y[:, 1]
        return len(output_items[0])
