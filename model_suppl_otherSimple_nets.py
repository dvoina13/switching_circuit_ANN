import torch
import torch.nn as nn


class CNN_simple_switch_const(nn.Module):
    def __init__(self, sz, vip):
        super(CNN_simple_switch_const, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout(p=0.2)
        #self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(16 * int((sz - 4)) * int((sz - 4)), 64)
        self.out = nn.Linear(64, 10)

        self.vip_lin1 = nn.Linear(vip, 16*int((sz-4))*int((sz-4)))
        self.vip_lin2 = nn.Linear(vip, 64)

        self.vip = vip
        self.constant = torch.ones(vip)

    def forward(self, x, sw, s_pr, device):

        sz = x.size()[2]
        out = self.cnn_1(x)
        #out = self.dropout2d(out)
        #out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        if sw==1:
            if s_pr == 0:
                if torch.cuda.is_available():
                    self.constant = self.constant.to(device)
                y1 = self.vip_lin1(self.constant.double())
                out = out+y1
            else:
                out = out*y1

        out = self.relu(out)

        out = self.fc1(out)
        #out = self.dropout(out)

        if sw == 1:
            if s_pr==0:
                y3 = self.vip_lin2(self.constant.double())
                out = out + y3
            else:
                out = out * y3

        out = self.relu(out)
        out = self.out(out)

        return out

# CNN with contextual input
class CNN_wInput(nn.Module):
    def __init__(self, sz):
        super(CNN_wInput, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout(p=0.2)
        #self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(16 * int((sz - 4)) * int((sz - 4)) + 1, 64)
        self.out = nn.Linear(65, 10)

    def forward(self, x, c):
        sz = 32
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        out = self.cnn_1(x)
        #out = self.dropout2d(out)
        #out = self.maxpool(out)

        out = out.view(out.size(0), -1)

        if c == 0:
            out = torch.cat((out, torch.zeros(out.size()[0], 1).to(device).double()), 1)
        elif c == 1:
            out = torch.cat((out, torch.ones(out.size()[0], 1).to(device).double()),1)

        out = self.relu(out)

        out = self.fc1(out)
        #out = self.dropout(out)
        out = self.relu(out)

        if c == 0:
            out = torch.cat((out, torch.zeros(out.size()[0], 1).double().to(device)), 1)
        elif c == 1:
            out = torch.cat((out, torch.ones(out.size()[0], 1).double().to(device)), 1)

        out = self.out(out)

        # return out
        return out


# CNN with contextual output
class CNN_wOutput(nn.Module):
    def __init__(self, sz):
        super(CNN_wOutput, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout(p=0.2)
        #self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(16 * int((sz - 4)) * int((sz - 4)), 64)
        self.out = nn.Linear(64, 11)

    def forward(self, x):
        sz = 32
        out = self.cnn_1(x)
        out = self.relu(out)
        #out = self.dropout2d(out)
        #out = self.maxpool(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        #out = self.dropout(out)

        out = self.out(out)

        # return out
        return out
