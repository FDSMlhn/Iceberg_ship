import torch
import torch.nn as nn
import math


class fcn(nn.Module):

    def __init__(self,layers=[64,128,256,256], num_classes=2,dropout= [0.2,0.2,0.3,0.7]):
        self.inplane=3
        super().__init__()
        self.layer1 = self._make_layer(layers[0], dropout=dropout[0])
        self.layer2 = self._make_layer(layers[1], dropout=dropout[1])
        self.layer3 = self._make_layer(layers[2], dropout=dropout[2])
        self.layer4 = self._make_layer(layers[3], dropout=dropout[3])
        

        self.cnn = nn.Conv2d(self.inplane, num_classes, kernel_size=3, padding=1,
                               bias=False)
        self.avg = nn.AvgPool2d(5)
        #self.dp = nn.Dropout(p=0.5)

    def _make_layer(self, planes, dropout):
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
        
        x = self.cnn(x)
        x = self.avg(x)
        
        return torch.squeeze(x)