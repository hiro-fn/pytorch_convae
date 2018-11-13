from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class AENet(nn.Module):
    def __init__(self):
        super(AENet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, stride=1),  # b, 16, 10, 10
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=1),  # b, 16, 5, 5
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, stride=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=1),  # b, 1, 28, 28
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x