from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        self.channel = channel
        self.reduction = reduction
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print("Shape after avgpool:",y.shape)
        # print("SEmoudel channel:",self.channel)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)

        return x * y.expand_as(x)
