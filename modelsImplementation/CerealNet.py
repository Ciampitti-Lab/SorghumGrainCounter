import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_after_ca = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x_after_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_after_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        
        # Apply spatial attention
        out = x_after_ca * sa
        
        return out

class CerealNet(nn.Module):
    def __init__(self, backboneModel=''):
        super(CerealNet, self).__init__()
        
        # Backbones
        self.backboneModel = backboneModel

        if self.backboneModel == "EfficientNetV2":
            lastLayers = 2
            backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
            self.backbone = nn.Sequential(*(list(backbone.children())[:-lastLayers]))
            reduced_channels = 1280
        
        if self.backboneModel == "ConvNeXt":
            lastLayers = 2
            backbone = torchvision.models.convnext_tiny(weights='DEFAULT')
            self.backbone = nn.Sequential(*(list(backbone.children())[:-lastLayers]))
            reduced_channels = 768
        
        if self.backboneModel == "MobileNetV3":
            lastLayers = 2
            backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            self.backbone = nn.Sequential(*(list(backbone.children())[:-lastLayers]))
            reduced_channels = 576
        
        if self.backboneModel == "MobileNetV3_Large":
            lastLayers = 2
            backbone = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
            self.backbone = nn.Sequential(*(list(backbone.children())[:-lastLayers]))
            reduced_channels = 960

        if self.backboneModel == "ResNeXt":
            lastLayers = 2
            backbone = torchvision.models.resnext50_32x4d(weights='DEFAULT')
            self.backbone = nn.Sequential(*(list(backbone.children())[:-lastLayers]))
            reduced_channels = 2048

        if self.backboneModel == "VGG16":
            lastLayers = 2
            backbone = torchvision.models.vgg16(weights='DEFAULT')
            self.backbone = nn.Sequential(*(list(backbone.children())[:-lastLayers]))
            reduced_channels = 512
            

        # CBAM
        self.cbam = CBAM(reduced_channels)

        # 1st Block
        self.first_block = nn.Sequential(
            nn.Conv2d(reduced_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor = 2)
        )

        # 2nd Block
        self.second_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor = 2)
        )

        # 3rd Block
        self.third_block = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor = 2)
        )

        # 4th Block
        self.fourth_block = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor = 2)
        )

        # 5th Block
        self.fifth_block = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2)
        )

    def forward(self, x):
        
        # Backbone
        backbone_out = self.backbone(x)

        # CBAM
        cbam_out = self.cbam(backbone_out)

        # 1st Block
        first_out = self.first_block(cbam_out)

        # 2nd Block
        second_out = self.second_block(first_out)
        
        # 3rd Block
        third_out = self.third_block(second_out)

        # 4th Block
        fourth_out = self.fourth_block(third_out)

        # 5th Block
        output = self.fifth_block(fourth_out)
        
        return output