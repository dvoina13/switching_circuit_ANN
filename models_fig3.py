import torch
import torch.nn as nn


class CNN_switch_2L(nn.Module):
    def __init__(self, sz, sp, vip):
        super(CNN_switch_2L, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout(p=0.2)
        #self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(16 * int((sz - 4)) * int((sz - 4)), 64)
        self.out = nn.Linear(64, 10)

        self.vip1 = nn.Conv2d(in_channels=16, out_channels=vip, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip2 = nn.Linear(64, vip)

        self.vip_back1 = nn.Conv2d(in_channels=vip, out_channels=16, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip_back2 = nn.Linear(vip, 64)

        self.vip = vip

    def forward(self, x, sw, s_pr):

        sz = x.size()[2]
        out = self.cnn_1(x)
        #out = self.dropout2d(out)
        #out = self.maxpool(out)

        #out = self.relu(out)
        if sw==1:
            if s_pr == 0:
                y1 = self.vip1(out)
                y1 = self.relu(y1)
                y1 = self.vip_back1(y1.view(-1, self.vip, int((sz-4)), int((sz-4))))
                #y1 = self.relu(y1)
                out = out+y1
            else:
                out = out*y1

        out = self.relu(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        #out = self.dropout(out)

        if sw == 1:
            if s_pr==0:
                y3 = self.vip2(out)
                y3 = self.relu(y3)
                y3 = self.vip_back2(y3)
                #y1 = self.relu(y1)
                out = out + y3
            else:
                out = out * y3

        out  = self.relu(out)
        out = self.out(out)

        return out


class CNN_switch_3L(nn.Module):
    def __init__(self, sz, sp, v1, v2):
        super(CNN_switch_3L, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout(p=0.2)
        #self.dropout2d = nn.Dropout2d(p=0.2)

        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(8 * int((sz - 8)) * int((sz - 8)), 64)
        self.out = nn.Linear(64, 10)

        self.vip1 = nn.Conv2d(in_channels=16, out_channels=v1, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip_back1 = nn.Conv2d(in_channels=v1, out_channels=16, kernel_size=sp, stride=1, padding=int((sp-1)/2))

        self.vip2 = nn.Conv2d(in_channels=8, out_channels=v1, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip_back2 = nn.Conv2d(in_channels=v1, out_channels=8, kernel_size=sp, stride=1, padding=int((sp-1)/2))

        self.vip3 = nn.Linear(64, v2)
        self.vip_back3 = nn.Linear(v2, 64)

        self.v1 = v1
        self.v2 = v2

    def forward(self, x, sw, s_pr):

        sz = x.size()[2]
        out = self.cnn_1(x)
        #out = self.dropout2d(out)
        #out = self.maxpool(out)

        if sw==1:
            if s_pr == 0:
                y1 = self.vip1(out)
                y1 = self.relu(y1)
                y1 = self.vip_back1(y1)
                out = out+y1
            else:
                out = out*y1

        out = self.relu(out)
        out = self.cnn_2(out)

        if sw==1:
            if s_pr == 0:
                y2 = self.vip2(out)
                y2 = self.relu(y2)
                y2 = self.vip_back2(y2)
                out = out+y2

        out = self.relu(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = self.dropout(out)

        if sw == 1:
            if s_pr==0:
                y3 = self.vip3(out)
                y3 = self.relu(y3)
                y3 = self.vip_back3(y3)
                out = out + y3
            else:
                out = out * y3

        out  = self.relu(out)
        out = self.out(out)

        return out

class CNN_switch_4L(nn.Module):
    def __init__(self, sz, sp, vip):
        super(CNN_switch_4L, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels= 16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(32 * int(((sz - 4) / 2 - 4) / 2) * int(((sz - 4) / 2 - 4) / 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

        self.vip1 = nn.Conv2d(in_channels=16, out_channels=vip, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip2 = nn.Conv2d(in_channels=32, out_channels=vip, kernel_size=5, stride=1, padding=int((sp-1)/2))
        self.vip3 = nn.Linear(128, 25)
        self.vip4 = nn.Linear(64, 10)

        self.vip_back1 = nn.Conv2d(in_channels=vip, out_channels=16, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip_back2 = nn.Conv2d(in_channels=vip, out_channels=32, kernel_size=sp, stride=1, padding=int((sp-1)/2))
        self.vip_back1 = nn.Linear(25, 128)
        self.vip_back2 = nn.Linear(10, 64)

        self.vip = vip
    def forward(self, x, sw, s_pr):

        sz = x.size()[2]

        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        if sw==1:
            if s_pr == 0:
                y1 = self.vip1(out)
                y1 = self.relu(y1)
                y1 = self.vip_back1(y1.view(-1, self.vip, int((sz-4)/2), int((sz-4)/2)))
                out = out+y1
            else:
                out = out*y1

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        if sw == 1:
            if s_pr == 0:
                y2 = self.vip2(out)
                y2 = self.relu(y2)
                y2 = self.vip_back2(y2.view(-1, self.vip, int((((sz - 4) / 2) - 4) / 2), int((((sz - 4) / 2) - 4) / 2)))
                out = out + y2
            else:
                out = out * y2

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)

        if sw == 1:
            if s_pr==0:
                y3 = self.vip3(out)
                y3 = self.relu(y3)
                y3 = self.vip_back3(y3)
                out = out + y3
            else:
                out = out * y3

        out = self.fc2(out)

        if sw == 1:
            if s_pr==0:
                y4 = self.vip4(out)
                y4 = self.relu(y4)
                y4 = self.vip_back4(y4)
                out = out + y4
            else:
                out = out * y4

        out = self.out(out)

        return out
