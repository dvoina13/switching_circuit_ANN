import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import os

from data_loader import load_training_validation_testing_data
from data_loader import load_training_testing_data
from models_fig5 import CNN_switch_2L_wMultipleOutputs

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, default=0.0, required=False, help="Set Transparency")
parser.add_argument("--vip", type=int, default=10, required=False, help="Set dimension of switch (filter)")
parser.add_argument("--sp", type=int, default=3, required=False, help="Set dimension of switch (space)")
parser.add_argument("--seed", type=int, default=1, required=False, help="seed")
parser.add_argument("--data_order", type=str, default="cifar-first", required=False,
                    help="MNIST+noise of MNIST+cifar first?")
parser.add_argument("--batch_size", type=int, default=50, required=False, help="batch size for training")
parser.add_argument("--test_batch_size", type=int, default=100, required=False, help="batch size for training")
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
parser.add_argument('--data-dir', type=str, default='/home/dvoina/CNNEx_project2/', metavar='P', help='path to data')
# parser.add_argument('--data-dir', type=str, default='/Users/dorisv/Desktop/some_scripts/CNNEx_project2/', metavar='P', help='path to data')
parser.add_argument('--save-path', type=str, default='./fig5_NNs', metavar='P', help='path to save final model')

args = parser.parse_args()

seed = args.seed
epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size

T = args.T
vip = args.vip
sp = args.sp

T_value = str(T)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# train_loader_noise, valid_loader_noise, test_loader_noise, train_loader_cifar, valid_loader_cifar, test_loader_cifar = load_training_validation_testing_data(
#        T_value, batch_size, test_batch_size, seed, args.data_dir, **kwargs)

train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar = load_training_testing_data(T_value, 0, 0,
                                                                                                          batch_size,
                                                                                                          test_batch_size,
                                                                                                          args.data_dir,
                                                                                                          **kwargs)

if args.data_order == "noise-first":
    train_loader_D1 = train_loader_noise
    # valid_loader_D1 = valid_loader_noise
    test_loader_D1 = test_loader_noise
    train_loader_D2 = train_loader_cifar
    # valid_loader_D2 = valid_loader_cifar
    test_loader_D2 = test_loader_cifar
    ord = "noise-then-cifar"
else:
    train_loader_D1 = train_loader_cifar
    # valid_loader_D1 = valid_loader_cifar
    test_loader_D1 = test_loader_cifar
    train_loader_D2 = train_loader_noise
    # valid_loader_D2 = valid_loader_noise
    test_loader_D2 = test_loader_noise
    ord = "cifar-then-noise"

save_path = args.save_path
save_path = '/'.join([args.save_path, ord])
if not os.path.exists(save_path):
    os.makedirs(save_path)

sz = 32

model = CNN_switch_2L_wMultipleOutputs(sz, int(sp), int(vip))

model.double()
if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
error = nn.CrossEntropyLoss()

O1_BSwitch = []
O2_BSwitch = []
Y1_Switch = []
Y2_Switch = []
O1_ASwitch = []
O2_ASwitch = []
O1_control = []
O2_control = []


def train(model, dataT, dataV):
    model.train()

    acc_test_duringL = []
    acc_train_duringL = []

    for batch_idx, (data, target) in enumerate(dataT):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, out1_bSwitch, out1_control, out1_aSwitch, out2_bSwitch, out2_control, out2_aSwitch, y1, y3 = model(data, switch, s_pr)

        predicted = torch.max(output.data, 1)[1]

        loss = error(output, target)

        loss.backward()
        optimizer.step()

        acc_train_duringL.append(float((predicted.to(device) == target).sum()) / len(target))

        """
        if batch_idx % 50 == 0:
            acc = validate(model, dataV)
            acc_test_duringL.append(acc)
        """
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(dataT.dataset),
        #        100. * batch_idx / len(dataT), loss.item()))

    return acc_train_duringL, acc_test_duringL


def validate(model, dataV, r):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    for data, target in dataV:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, out1_bSwitch, out1_control, out1_aSwitch, out2_bSwitch, out2_control, out2_aSwitch, y1, y2 = model(data, switch, s_pr)
        # sum up batch loss

        if r==1:
            O1_BSwitch.append(out1_bSwitch)
            O2_BSwitch.append(out1_bSwitch)
            Y1_Switch.append(y1)
            Y2_Switch.append(y2)
            O1_ASwitch.append(out1_aSwitch)
            O2_ASwitch.append(out2_bSwitch)
            O1_control.append(out1_control)
            O2_control.append(out2_control)

        val_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        total += len(target)

    val_loss /= total
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total,
        100. * correct / total))

    if r==0:
        return 100. * correct / total
    else:
        return 100. * correct / total, O1_BSwitch, O2_BSwitch, O1_ASwitch, O2_ASwitch, O1_control, O2_control, Y1_Switch, Y2_Switch


s_pr = 0
switch = 0

for name, param in model.named_parameters():
    if ('vip' in name):
        param.requires_grad = False

for name, param in model.named_parameters():
    print(name, param.requires_grad)

for epoch in range(1, epochs + 1):
    print(epoch)
    [accTrain, accTest] = train(model, train_loader_D1, test_loader_D1)

acc_initial_D1 = validate(model, test_loader_D1, 0)
acc_initial_D2 = validate(model, test_loader_D2, 0)
print("accuracy of testing the first dataset before switch is on {}%".format(acc_initial_D1))
print("accuracy of testing the second dataset before switch is on {}%".format(acc_initial_D2))

s_pr = 0
switch = 1

for name, param in model.named_parameters():
    if ('cnn' in name) | ('fc' in name) | ('out' in name):
        param.requires_grad = False
    else:
        param.requires_grad = True

for name, param in model.named_parameters():
    print(name, param.requires_grad)

for epoch in range(1, epochs + 1):
    print(epoch)
    [accTrain, accTest] = train(model, train_loader_D2, test_loader_D2)

acc_afterT_D2, O1_BSwitch, O2_BSwitch, O1_ASwitch, O2_ASwitch, O1_control, O2_control, Y1_Switch, Y2_Switch = validate(model, test_loader_D2, 1)
print("accuracy of testing the second dataset after switch is on {}%".format(acc_afterT_D2))

del train_loader_D1, test_loader_D1, train_loader_D2, test_loader_D2, accTrain, accTest

O1_ASwitch = torch.stack(O1_ASwitch)
O2_ASwitch = torch.stack(O2_ASwitch)
O1_control = torch.stack(O1_control)
O2_control = torch.stack(O2_control)

O1_ASwitch_view = O1_ASwitch.view(-1,16,28,28)
O1_control_view = O1_control.view(-1,16,28,28)
O2_ASwitch_view = O2_ASwitch.view(-1,64)
O2_control_view = O2_control.view(-1,64)

O1_control_sparsity = len(np.where(O1_control_view.cpu().numpy().flatten() == 0)[0])
O1_ASwitch_sparsity = len(np.where(O1_ASwitch_view.detach().cpu().numpy().flatten() == 0)[0])

print("layer 1 sparsity: ", O1_control_sparsity, O1_ASwitch_sparsity)

O2_control_sparsity = len(np.where(O2_control_view.detach().cpu().numpy().flatten() == 0)[0])
O2_ASwitch_sparsity = len(np.where(O2_ASwitch_view.detach().cpu().numpy().flatten() == 0)[0])

print("layer 2 sparsity: ", O2_control_sparsity, O2_ASwitch_sparsity)


del O1_BSwitch, O1_ASwitch, O1_control, O2_BSwitch, O2_ASwitch, O2_control

Y1_Switch = torch.stack(Y1_Switch)
Y2_Switch = torch.stack(Y2_Switch)

Y1_Switch_mean = torch.mean(Y1_Switch)
Y1_Switch_std = torch.std(Y1_Switch)
Y1_Switch_stats = np.array([Y1_Switch_mean, Y1_Switch_std])
Y2_Switch_mean = torch.mean(Y2_Switch)
Y2_Switch_std = torch.std(Y2_Switch)
Y2_Switch_stats = np.array([Y2_Switch_mean, Y2_Switch_std])

print("layer 1 switching unit mean and std contribution: ", Y1_Switch_stats)

save_path2 = save_path + '/O1_ASwitch_' + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed-" + str(
    seed) + ".npy"
torch.save(O1_ASwitch_sparsity, save_path2)
save_path3 = save_path + "/O1_control_" + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed" + str(seed) + ".npy"
torch.save(O1_control_sparsity, save_path3)
save_path4 = save_path + "/Y1_Switch_" + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed" + str(seed) + ".npy"
torch.save(Y1_Switch_stats, save_path4)

save_path2 = save_path + '/O2_ASwitch_' + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed-" + str(
    seed) + ".npy"
torch.save(O2_ASwitch_sparsity, save_path2)
save_path3 = save_path + "/O2_control_" + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed" + str(seed) + ".npy"
torch.save(O2_control_sparsity, save_path3)
save_path4 = save_path + "/Y2_Switch_" + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed" + str(seed) + ".npy"
torch.save(Y2_Switch_stats, save_path4)

