import torch
import torch.nn as nn

class CNN_w2L_main(nn.Module):
    def __init__(self, sz):
        super(CNN_w2L_main, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(16 * (sz - 4) * (sz - 4), 64)
        self.out = nn.Linear(64,10)

    def forward(self, x):
        sz = x.size()[2]

        out = self.cnn_1(x)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)

        out = self.out(out)

        return out

class CNN_w3L_LargeWidth(nn.Module):
        def __init__(self, sz):
            super(CNN_w3L_LargeWidth, self).__init__()

            self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=5, stride=1, padding=0)
            self.relu = nn.ReLU()

            self.fc1 = nn.Linear(30 * (sz - 4) * (sz - 4), 300)
            self.fc2 = nn.Linear(300,50)
            self.out = nn.Linear(50,10)
            
        def forward(self, x):
            sz = x.size()[2]

            out = self.cnn_1(x)
            out = self.relu(out)

            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            
            out = self.fc2(out)
            out = self.relu(out)
            
            out = self.out(out)

            return out
            
class CNN_4L_wDMP(nn.Module):
    def __init__(self, sz):
        super(CNN_4L_wDMP, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(32 * int(((sz - 4) / 2 - 4) / 2) * int(((sz - 4) / 2 - 4) / 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):

        sz = x.size()[2]

        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)

        out = self.fc2(out)

        out = self.out(out)

        return out

class CNN_2L_wDMP(nn.Module):
    def __init__(self, sz):
        super(CNN_2L_wDMP, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(16 * int((sz - 4) / 2) * int((sz - 4) / 2), 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):

        sz = x.size()[2]

        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)

        out = self.out(out)

        return out

class CNN_3LL(nn.Module):
    def __init__(self, sz):
        super(CNN_3LL, self).__init__()

        self.fc1 = nn.Linear(32 * 32, 150)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(150, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out, out_layer1, out_layer2, out_layer3, out_layer4

class CNN_2LL(nn.Module):
    def __init__(self, sz):
        super(CNN_2LL, self).__init__()

        self.fc1 = nn.Linear(32 * 32, 100)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):

        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        return out
