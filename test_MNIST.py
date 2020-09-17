import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter       
writer = SummaryWriter()

class MNISTClass(nn.Module):
    def __init__(self):
        super(MNISTClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1080, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # conv1(kernel=3, filters=15) 28x28x1 -> 26x26x15
        x = F.relu(self.conv1(x))

        # conv2(kernel=3, filters=20) 26x26x15 -> 13x13x30
        # max_pool(kernel=2) 13x13x30 -> 6x6x30
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))

        # flatten 6x6x30 = 1080
        x = x.view(-1, 1080)

        # 1080 -> 100
        x = F.relu(self.fc1(x))

        # 100 -> 10
        x = self.fc2(x)

        # transform to logits
        return F.log_softmax(x, dim=1)


# Record training loss from each epoch into the writer
    writer.add_scalar(\\'Train/Loss\\', loss.item(), epoch)
    writer.flush()
    
        
    # Record loss and accuracy from the test run into the writer
    writer.add_scalar(\\'Test/Loss\\', test_loss, epoch)
    writer.add_scalar(\\'Test/Accuracy\\', accuracy, epoch)
    writer.flush()