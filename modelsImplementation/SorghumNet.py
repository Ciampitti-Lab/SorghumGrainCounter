import torch
import torch.nn as nn

class SorghumNet(nn.Module):
    def __init__(self, input_channels=3):
        super(SorghumNet, self).__init__()
        
        # Up
        self.up_branch = nn.Sequential(
            nn.Conv2d(input_channels, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(80, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(160, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )

        # Middle
        self.middle_branch = nn.Sequential(
            nn.Conv2d(input_channels, 40, kernel_size=5, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(40, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(80, 40, kernel_size=5, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        
        # Down
        self.bottom_branch = nn.Sequential(
            nn.Conv2d(input_channels, 20, kernel_size=7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 40, kernel_size=7, padding=3),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(40, 20, kernel_size=7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=7, padding=3),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        
        # Upsampling and output
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.output_conv = nn.Conv2d(100, 1, kernel_size=1, padding=0)
        
    
    def forward(self, x):
        
        # Up branch
        up = self.up_branch(x)
        
        # Middle branch
        middle = self.middle_branch(x)

        # Bottom branch
        bottom = self.bottom_branch(x)
        
        # Concatenate feature maps
        x_concat = torch.cat([up, middle, bottom], dim=1)
        
        # Upsample and output
        x_upsampled = self.upsample(x_concat)
        output = self.output_conv(x_upsampled)
        
        return output