import torch
import torch.nn as nn


class CNN_switch_2L_wMultipleOutputs(nn.Module):
    def __init__(self, sz, sp, vip):
        super(CNN_switch_2L_wMultipleOutputs, self).__init__()

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

        y1 = None; y3 = None; 
        sz = x.size()[2]
        out = self.cnn_1(x)
        #out = self.dropout2d(out)
        #out = self.maxpool(out)

        out1_bSwitch = out

        #out = self.relu(out)
        if sw==1:
            if s_pr == 0:
                y1 = self.vip1(out)
                y1 = self.relu(y1)
                y1 = self.vip_back1(y1.view(-1, self.vip, int((sz-4)), int((sz-4))))
                out = out+y1
            else:
                out = out*y1

        out1_control = self.relu(out1_bSwitch)
        out1_aSwitch = self.relu(out)

        out = self.relu(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        #out = self.dropout(out)

        out2_bSwitch = out

        if sw == 1:
            if s_pr==0:
                y3 = self.vip2(out)
                y3 = self.relu(y3)
                y3 = self.vip_back2(y3)
                out = out + y3
            else:
                out = out * y3

        out2_control = self.relu(out2_bSwitch)
        out2_aSwitch = self.relu(out)

        out  = self.relu(out)
        out = self.out(out)

        return out, out1_bSwitch, out1_control, out1_aSwitch, out2_bSwitch, out2_control, out2_aSwitch, y1, y3
