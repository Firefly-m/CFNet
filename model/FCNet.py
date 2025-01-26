# FCNet
# 2024.7.2
# D.Mo

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import sys
sys.path.append('./libs')
from CFF import CFF
from CAM import CAM
from FRM import FRM
from FIAD import FIAD
sys.path.append('./libs/backbone')
from pvtv2 import pvt_v2_b2

class FCNet(nn.Module):
    def __init__(self, mode='train'):
        super().__init__()

        # backbone
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pth/backbone/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.CFF = CFF(512, 320, 128)
        # Collaborative feature strengthening module
        self.CAM = CAM(256, 64)

        # Collaborative induced foreground aggregation module
        self.FRM = FRM(256, 512, 320, 128, 64, 64) 
        self.FIAD = FIAD(64, 64)


    def forward(self, input):
        N, _, H, W = input.size()

        # backbone
        pvt = self.backbone(input)
        x1 = pvt[0]     # 10, 64, 64, 64
        x2 = pvt[1]     # 10, 128, 32, 32
        x3 = pvt[2]     # 10, 320, 16, 16
        x4 = pvt[3]     # 10, 512, 8, 8

        fusion = self.CFF(x4, x3, x2)            # Feature Fusion
        gr = self.CAM(fusion)                    # Consensus Awareness Module 
        fr = self.FRM(fusion, x4, x3, x2, x1)    # Foreground Refinement Module 
        result = self.FIAD(gr, fr)               # Foreground-induced Integrity Aggregation Decoder 
        result = F.interpolate(result, size=(H, W), mode='bilinear', align_corners=False)

        # bs, 1, H, W
        return result

if __name__ == '__main__':
    # Init model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = FCNet()
    model = model.to(device)

    images = torch.randn(2, 3, 256, 256)
    images = images.to(device)

    S_ge = model(images)


