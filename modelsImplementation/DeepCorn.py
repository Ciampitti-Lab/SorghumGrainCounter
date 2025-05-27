import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class DeepCorn(nn.Module):
    def __init__(self, input_channels=3):
        super(DeepCorn, self).__init__()

        # Backbone (Truncated VGG-16)
        self.backbone_conv1 = nn.Sequential(
            # Conv-3-64
            nn.Conv2d(input_channels, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # Conv-3-64
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # S2 pooling
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.backbone_conv2 = nn.Sequential(
            # Conv-3-128
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # Conv-3-128
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # S2 pooling
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.backbone_conv3 = nn.Sequential(
            # Conv-3-256
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # Conv-3-256
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # Conv-3-256
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            # S2 pooling
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.backbone_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.backbone_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )

        # Up layers
        self.up_1 = nn.Conv2d(512, 1024, kernel_size=1, padding='same')
        self.up_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
        self.up_3 = nn.Conv2d(1024, 256, kernel_size=1, padding='same')
        self.up_4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=2)
        self.up_5 = nn.Conv2d(512, 128, kernel_size=1, padding='same')
        self.up_6 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=2)
        self.up_7 = nn.Conv2d(256, 128, kernel_size=1, padding='same')
        self.up_8 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=2)
        self.up_9 = nn.Conv2d(256, 128, kernel_size=1, padding='same')
        self.up_10 = nn.Conv2d(128, 256, kernel_size=3, padding='same')

        # After concat
        self.after_1 = nn.Conv2d(2816, 1024, kernel_size=1, padding='same')
        self.after_2 = nn.Conv2d(1024, 256, kernel_size=3, padding='same')
        self.after_3 = nn.Conv2d(256, 1, kernel_size=1)

        # Interpolation
        self.interpolation = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self._initialize_weights()
        self._load_pretrained_vgg()

    def _initialize_weights(self):
        def real_init_weights(m):
            if isinstance(m, nn.Conv2d):    
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Apply to all modules
        for m in self.modules():
            real_init_weights(m)
    
    def _load_pretrained_vgg(self):
        # Load official VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Conv1 block (features[0:4] in original VGG: Conv->ReLU->Conv->ReLU->Pool)
        self.backbone_conv1.load_state_dict({
            '0.weight': vgg[0].weight,
            '0.bias': vgg[0].bias,
            '2.weight': vgg[2].weight,
            '2.bias': vgg[2].bias
        })
        
        # Conv2 block (features[5:9] in original VGG)
        self.backbone_conv2.load_state_dict({
            '0.weight': vgg[5].weight,
            '0.bias': vgg[5].bias,
            '2.weight': vgg[7].weight,
            '2.bias': vgg[7].bias
        })
        
        # Conv3 block (features[10:16] in original VGG: 3 conv layers)
        self.backbone_conv3.load_state_dict({
            '0.weight': vgg[10].weight,
            '0.bias': vgg[10].bias,
            '2.weight': vgg[12].weight,
            '2.bias': vgg[12].bias,
            '4.weight': vgg[14].weight,
            '4.bias': vgg[14].bias
        })
        
        # Conv4 block (features[17:23] in original VGG)
        self.backbone_conv4.load_state_dict({
            '0.weight': vgg[17].weight,
            '0.bias': vgg[17].bias,
            '2.weight': vgg[19].weight,
            '2.bias': vgg[19].bias,
            '4.weight': vgg[21].weight,
            '4.bias': vgg[21].bias
        })
        
        # Conv5 block (features[24:30] in original VGG - no final pooling)
        self.backbone_conv5.load_state_dict({
            '0.weight': vgg[24].weight,
            '0.bias': vgg[24].bias,
            '2.weight': vgg[26].weight,
            '2.bias': vgg[26].bias,
            '4.weight': vgg[28].weight,
            '4.bias': vgg[28].bias
        })
    
    def forward(self, x):
        
        # Backbone feature extraction
        x1 = self.backbone_conv1(x)
        x2 = self.backbone_conv2(x1)
        x3 = self.backbone_conv3(x2)
        x4 = self.backbone_conv4(x3)
        x5 = self.backbone_conv5(x4)

        # Up layers
        up_1 = self.up_1(x5)
        up_2 = self.up_2(up_1)
        up_3 = self.up_3(up_2)
        up_4 = self.up_4(up_3)
        up_5 = self.up_5(up_4)
        up_6 = self.up_6(up_5)
        up_7 = self.up_7(up_6)
        up_8 = self.up_8(up_7)
        up_9 = self.up_9(up_8)
        up_10 = self.up_10(up_9)

        # Concat
        up_2 = F.interpolate(up_2, size=x4.shape[2:], mode='bilinear', align_corners=True)
        up_4 = F.interpolate(up_4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        up_6 = F.interpolate(up_6, size=x4.shape[2:], mode='bilinear', align_corners=True)
        up_8 = F.interpolate(up_8, size=x4.shape[2:], mode='bilinear', align_corners=True)
        up_10 = F.interpolate(up_10, size=x4.shape[2:], mode='bilinear', align_corners=True)

        concatenated = torch.cat([x4, up_2, up_4, up_6, up_8, up_10], dim=1)
        
        # After Concat
        after_1 = self.after_1(concatenated)
        after_2 = self.after_2(after_1)
        after_3 = self.after_3(after_2)

        # Interpolation
        density_map = self.interpolation(after_3)
        
        return density_map