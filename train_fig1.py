import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import os

from data_loader import load_training_testing_data
from models_fig1 import CNN_w2L_main, CNN_w3L_LargeWidth

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, default=0.0, required=False, help="Set Transparency")
parser.add_argument("--T2_use", type=int, default=0, required=False, help="Binary value showing is we are using 2nd transparency")
parser.add_argument("--T2", type=float, default=0.5, required=False, help="Set 2nd value for Transparency (for second dataset)")
parser.add_argument("--seed", type=int, default=1, required=False, help="seed")

parser.add_argument("--data_order", type=str, default="cifar-first", required=False, help="MNIST+noise of MNIST+cifar first?")
parser.add_argument("--batch_size", type=int, default=50, required=False, help="batch size for training")
parser.add_argument("--test_batch_size", type=int, default=1000, required=False, help="batch size for training")
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
parser.add_argument('--data-dir', type=str, default='/home/dvoina/CNNEx_project2/', metavar='P', help='path to data')
parser.add_argument('--save-path', type=str, default='./fig1_NNs', metavar='P', help='path to save final model')

args = parser.parse_args()

seed = args.seed
epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size

T = args.T
T2 = args.T2
T2_use = args.T2_use

T_value = str(T)
T2_value = str(T2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if T2_use == 0:
    train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar = load_training_testing_data(
        T_value, T2_use, T2_value, batch_size, test_batch_size, args.data_dir, **kwargs)
else:
    train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar, train_loader_noise2, test_loader_noise2, train_loader_cifar2, test_loader_cifar2 = load_training_testing_data(
        T_value, T2_use, T2_value, batch_size, test_batch_size, args.data_dir, **kwargs)

if args.data_order == "noise-first":
    train_loader_D1 = train_loader_noise
    test_loader_D1 = test_loader_noise
    train_loader_D2 = train_loader_cifar
    test_loader_D2 = test_loader_cifar
    ord = "noise-then-cifar_LargeWidth"
else:
    train_loader_D1 = train_loader_cifar
    test_loader_D1 = test_loader_cifar
    train_loader_D2 = train_loader_noise
    test_loader_D2 = test_loader_noise
    ord = "cifar-then-noise_LargeWidth"

sz=32

#model = CNN_w2L_main(sz)
model = CNN_w3L_LargeWidth(sz)

model.double()
if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
error = nn.CrossEntropyLoss()

def train(model, dataT):
    model.train()
    for batch_idx, (data, target) in enumerate(dataT):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = error(output, target)

        loss.backward()
        optimizer.step()

        #if batch_idx % 100 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(dataT.dataset),
        #        100. * batch_idx / len(dataT), loss.item()))


def validate(model, dataV):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    for data, target in dataV:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        val_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        total += len(target)

    val_loss /= total
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total,
        100. * correct / total))

    return 100. * correct / total

for epoch in range(1,epochs+1):
    print(epoch)
    train(model, train_loader_D1)
    validate(model, test_loader_D1)

acc_valid_D1 = validate(model, test_loader_D1)
print('Test accuracy for dataset 1: {}%'.format(acc_valid_D1))
acc_valid_D2_initial = validate(model, test_loader_D2)
print('Test accuracy for dataset 2 (before training): {}%'.format(acc_valid_D2_initial))

for epoch in range(1,epochs+1):
    print(epoch)
    train(model, train_loader_D2)
    validate(model, test_loader_D2)

acc_valid_D2_after = validate(model, test_loader_D2)
print('Test accuracy for dataset 2 (after training): {}%'.format(acc_valid_D2_after))

acc_valid_D1_after_CF = validate(model, test_loader_D1)
print('Test accuracy for dataset 1 (to test catastrophic forgetting): {}%'.format(acc_valid_D1_after_CF))

save_path = args.save_path
save_path =  '/'.join([args.save_path, ord])
if not os.path.exists(save_path):
        os.makedirs(save_path)

save_path1 = save_path + '/initial_D1_' + T_value + "-seed" + str(seed) + ".npy"
torch.save(acc_valid_D1, save_path1)
save_path2 = save_path + '/initial_D2_' + T_value + "-seed" + str(seed) + ".npy"
torch.save(acc_valid_D2_initial, save_path2)
save_path3 = save_path +  "/afterT_D2_" + T_value + "-seed" + str(seed) + ".npy"
torch.save(acc_valid_D2_after, save_path3)
save_path4 = save_path + "/afterT_D1_checkCF_" + T_value + "-seed" + str(seed) + ".npy"
torch.save(acc_valid_D1_after_CF, save_path4)


