import torch.nn as nn
import math


class plain_cnn(nn.Module):

    def __init__(self,layers=[64,128,128,64], num_classes=2,dropout= [0.2,0.2,0.3,0.3]):
        self.inplane=3
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(64, layers[0], dropout=dropout[0])
        self.layer2 = self._make_layer(128, layers[1], dropout=dropout[1])
        self.layer3 = self._make_layer(128, layers[2], dropout=dropout[2])
        self.layer4 = self._make_layer(64, layers[3], dropout=dropout[3])

        self.fc1 = nn.Linear(64 * 7*7, 64 * 7*7)
        self.bn1 = nn.BatchNorm1d(64 * 7*7)
        self.relu1 =nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)
        
        self.fc2 = nn.Linear(32 * 7*7, 32 * 7*7)
        self.bn2 = nn.BatchNorm1d(32 * 7*7)
        self.relu2 =nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.2)
        
        self.fc3 = nn.Linear(32 * 7*7, num_classes)
        #self.dp = nn.Dropout(p=0.5)

    def _make_layer(self, planes, blocks, dropout):
        layer = nn.Sequential(
                    nn.Conv2d(self.inplane, planes, kernel_size=3, padding=1,
                               bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                    nn.Dropout(dropout)
                            )
        self.inplane=planes
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x
