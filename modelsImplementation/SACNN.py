import torch
import torch.nn as nn
import torch.nn.functional as F

class SACNN(nn.Module):
    def __init__(self, input_channels=3):
        super(SACNN, self).__init__()
        
        # Block 1
        self.block1_conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 5
        self.sacnn5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.sacnn5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.sacnn5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.sacnn5_pool = nn.MaxPool2d(3, stride=1, padding=1)
        
        # Block 6
        self.sacnn6_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        
        # Block 7 (Final layers)
        self.sacnn7_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.sacnn7_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.sacnn7_conv3 = nn.Conv2d(256, 1, kernel_size=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, x):
        # Block 1
        x11 = F.relu(self.block1_conv1(x))
        x12 = F.relu(self.block1_conv2(x11))
        xp1 = self.block1_pool(x12)
        
        # Block 2
        x21 = F.relu(self.block2_conv1(xp1))
        x22 = F.relu(self.block2_conv2(x21))
        xp2 = self.block2_pool(x22)
        
        # Block 3
        x31 = F.relu(self.block3_conv1(xp2))
        x32 = F.relu(self.block3_conv2(x31))
        x33 = F.relu(self.block3_conv3(x32))
        x34 = F.relu(self.block3_conv4(x33))
        xp3 = self.block3_pool(x34)
        
        # Block 4
        x41 = F.relu(self.block4_conv1(xp3))
        x42 = F.relu(self.block4_conv2(x41))
        x43 = F.relu(self.block4_conv3(x42))
        xp4 = self.block4_pool(x43)
        
        # Block 5
        x51 = F.relu(self.sacnn5_conv1(xp4))
        x52 = F.relu(self.sacnn5_conv2(x51))
        x53 = F.relu(self.sacnn5_conv3(x52))
        xp5 = self.sacnn5_pool(x53)
        
        # Block 6
        x61 = F.relu(self.sacnn6_conv1(xp5))
        
        # Merge x53 and x61
        x6153 = torch.cat([x53, x61], dim=1)
        
        # First deconvolution
        xd = F.relu(self.deconv1(x6153))
        
        # Merge with x43
        xd43 = torch.cat([xd, x43], dim=1)
        xd43 = F.relu(self.deconv2(xd43))
        
        # Final block
        x71 = F.relu(self.sacnn7_conv1(xd43))
        x72 = F.relu(self.sacnn7_conv2(x71))
        x73 = F.relu(self.sacnn7_conv3(x72))
        output = self.upsample(x73)
        return output