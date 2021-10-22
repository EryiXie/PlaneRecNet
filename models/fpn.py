import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg

class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        start_level=0
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = cfg.fpn.num_features
        self.num_ins = len(in_channels)

        self.backbone_end_level = self.num_ins
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()


        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], self.out_channels, kernel_size=1)
            self.lateral_convs.append(l_conv)
        
        for ii in range(self.start_level, self.backbone_end_level):
            p_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
            self.fpn_convs.append(p_conv)

        if cfg.fpn.high_level_mode == 'retina':
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=2)
                for _ in range(2)
            ])

        self.interpolation_mode     = cfg.fpn.interpolation_mode
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers
        self.high_level_mode        = cfg.fpn.high_level_mode
        assert not(cfg.fpn.high_level_mode == 'retina' and cfg.fpn.high_level_mode == 'original' and (cfg.fpn.high_level_mode is not None))
            
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        x = torch.zeros(1, device=inputs[0].device)
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i > 0:
                _, _, h, w = inputs[i + self.start_level].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x = lateral_conv(inputs[i + self.start_level]) + x
            laterals.append(x)

        # Top-down
        outs = []
        for ii, predict_conv in enumerate(self.fpn_convs):
            outs.append(predict_conv(laterals[ii]))
            if self.relu_pred_layers:
                F.relu(outs[ii], inplace=True)
        
        if self.high_level_mode == 'original':
            p6 = F.max_pool2d(outs[-1], kernel_size=1, stride=2, padding=0)
            outs.append(p6)
        elif self.high_level_mode == 'retina':
            p6 = self.downsample_layers[0](outs[-1])
            p7 = self.downsample_layers[1](F.relu(p6, inplace=True))
            outs.append(p6)
            outs.append(p7)
        return outs