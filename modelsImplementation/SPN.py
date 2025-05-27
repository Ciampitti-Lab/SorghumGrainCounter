import torch.nn as nn
import torch
from torchvision import models

def create_conv2d_block(in_channels, kernel_size, n_filter, dilated_rate=1, batch_norm=True):
    # padding formula  https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
    """
    o = output
    p = padding
    k = kernel_size
    s = stride
    d = dilation
    """
#     o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    k = kernel_size
    d = dilated_rate
    padding_rate = int((k + (k-1)*(d-1))/2)
    conv2d =  nn.Conv2d(in_channels, n_filter, kernel_size, padding=padding_rate, dilation = dilated_rate)
    bn = nn.BatchNorm2d(n_filter)
    relu = nn.ReLU(inplace=True)
    if batch_norm:
        return [conv2d, bn, relu]
    else:
        return [conv2d, relu]

class ScalePyramidModule(nn.Module):
    def __init__(self):
        super(ScalePyramidModule, self).__init__()
        self.a = nn.Sequential(*create_conv2d_block(512, 3, 512, 2))
        self.b = nn.Sequential(*create_conv2d_block(512, 3, 512, 4))
        self.c = nn.Sequential(*create_conv2d_block(512, 3, 512, 8))
        self.d = nn.Sequential(*create_conv2d_block(512, 3, 512, 12))
    def forward(self,x):
        xa = self.a(x)
        xb = self.b(x)
        xc = self.c(x)
        xd = self.d(x)
        return torch.cat((xa, xb, xc, xd), 1)

def make_layers_by_cfg(cfg, in_channels = 3,batch_norm=False, dilation = True):
    """
    cfg: list of tuple (number of layer, kernel, n_filter, dilated) or 'M'
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # number of layer, kernel, n_filter, dilated
            for t in range(v[0]):
                layers += create_conv2d_block(in_channels, v[1], v[2], v[3], batch_norm = batch_norm)
                in_channels = v[2]
    return nn.Sequential(*layers) 

class SPN(nn.Module):
    def __init__(self, load_weights=False):
        super(SPN, self).__init__()
        self.frontend_config = [(2,3,64,1), 'M', (2,3,128,1), 'M', (3,3,256,1), 'M', (3,3,512,1)] 
        self.backend_config = [(1,3,256,1), (1,3,512,1)]
        self.frontend = make_layers_by_cfg(self.frontend_config)
        self.spm = ScalePyramidModule()
        self.backend = make_layers_by_cfg(self.backend_config, 512*4, batch_norm=True)
        self.output_layer = nn.Sequential(*create_conv2d_block(512, 1, 1, 1))
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(list(self.frontend.state_dict().items()))):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x1 = self.frontend(x)
        x2 = self.spm(x1)
        x3 = self.backend(x2)
        x4 = self.output_layer(x3)
        output = self.upsample(x4)
        return output