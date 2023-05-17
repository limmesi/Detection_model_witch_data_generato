import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        bbox_cords = self.bbox_regressor(x)
        return bbox_cords


