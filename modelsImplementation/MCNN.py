import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(MCNN, self).__init__()
        
        # Up branch
        self.up_brach = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
        # Middle branch
        self.middle_brach = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Down branch
        self.down_brach = nn.Sequential(
            nn.Conv2d(input_channels, 20, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        
        # Output layers
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(8 + 10 + 12, 1, kernel_size=1, padding=0)

    def forward(self, x):

        # Up branch
        up = self.up_brach(x)
        
        # Middle branch
        middle = self.middle_brach(x)
        
        # Down branch
        down = self.down_brach(x)
        
        # Concatenate all branches
        output = torch.cat([up, middle, down], dim=1)
        
        # Upsample and final conv
        output = self.upsample(output)
        output = self.final_conv(output)
        
        return output