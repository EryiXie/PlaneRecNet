import torch
import torch.nn as nn
from models.dcn import DeformableConv2d

class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        norm_layer=nn.BatchNorm2d,
        dilation=1,
        use_dcn=False,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn1 = norm_layer(planes)
        if use_dcn:
            self.conv2 = DeformableConv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=True
                )
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=False,
                dilation=dilation,
            )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False, dilation=dilation
        )
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(
        self,
        layers,
        dcn_layers=[0, 0, 0, 0],
        dcn_interval=1,
        atrous_layers=[],
        block=Bottleneck,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(
            block, 
            64, 
            layers[0], 
            dcn_layers=dcn_layers[0], 
            dcn_interval=dcn_interval
        )
        self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dcn_layers=dcn_layers[1],
            dcn_interval=dcn_interval,
        )
        self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dcn_layers=dcn_layers[2],
            dcn_interval=dcn_interval,
        )
        self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dcn_layers=dcn_layers[3],
            dcn_interval=dcn_interval,
        )

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(
        self, block, planes, blocks, stride=1, dcn_layers=0, dcn_interval=1
    ):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    dilation=self.dilation,
                ),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        use_dcn = dcn_layers >= blocks
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.norm_layer,
                self.dilation,
                use_dcn=use_dcn,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_dcn = ((i + dcn_layers) >= blocks) and (i % dcn_interval == 0)
            layers.append(
                block(
                    self.inplanes, planes, norm_layer=self.norm_layer, use_dcn=use_dcn
                )
            )
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers: # for c2 to c5
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith("layer"):
                idx = int(key[5])
                new_key = "layers." + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(
            block, conv_channels // block.expansion, blocks=depth, stride=downsample
        )


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    backbone = cfg.type(*cfg.args)

    # Add downsampling layers until we reach the number we need
    num_layers = max(cfg.selected_layers) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone
