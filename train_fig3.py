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
from data_loader import load_training_testing_data_permuted
from models_fig3 import CNN_switch_2L, CNN_switch_3L

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, default=0.0, required=False, help="Set Transparency")
parser.add_argument("--vip", type=int, default=5, required=False, help="Set dimension of switch (filter)")
parser.add_argument("--sp", type=int, default=5, required=False, help="Set dimension of switch (space)")
parser.add_argument("--seed", type=int, default=1, required=False, help="seed")
parser.add_argument("--data_order", type=str, default="cifar-first", required=False, help="MNIST+noise of MNIST+cifar first?")
parser.add_argument("--batch_size", type=int, default=50, required=False, help="batch size for training")
parser.add_argument("--test_batch_size", type=int, default=1000, required=False, help="batch size for training")
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
parser.add_argument('--data-dir', type=str, default='/home/dvoina/CNNEx_project2/', metavar='P', help='path to data')
#parser.add_argument('--data-dir', type=str, default='/Users/dorisv/Desktop/some_scripts/CNNEx_project2/', metavar='P', help='path to data')
parser.add_argument('--save-path', type=str, default='./fig3_NNs', metavar='P', help='path to save final model')
parser.add_argument('--saved', type=int, default=0, metavar='P', help='save model?')

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

#train_loader_noise, valid_loader_noise, test_loader_noise, train_loader_cifar, valid_loader_cifar, test_loader_cifar = load_training_validation_testing_data(
#        T_value, batch_size, test_batch_size, seed, args.data_dir, **kwargs)

train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar = load_training_testing_data(T_value, 0, 0, batch_size, test_batch_size, args.data_dir, **kwargs)

#train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar = load_training_testing_data_permuted(T_value, batch_size, test_batch_size, args.data_dir, **kwargs)

if args.data_order == "noise-first":
    train_loader_D1 = train_loader_noise
    #valid_loader_D1 = valid_loader_noise
    test_loader_D1 = test_loader_noise
    train_loader_D2 = train_loader_cifar
    #valid_loader_D2 = valid_loader_cifar
    test_loader_D2 = test_loader_cifar
    ord = "noise-then-cifar_3L_all_unpermuted"
else:
    train_loader_D1 = train_loader_cifar
    #valid_loader_D1 = valid_loader_cifar
    test_loader_D1 = test_loader_cifar
    train_loader_D2 = train_loader_noise
    #valid_loader_D2 = valid_loader_noise
    test_loader_D2 = test_loader_noise
    ord = "cifar-then-noise_3L_all_unpermuted"

save_path = args.save_path
save_path =  '/'.join([args.save_path, ord])
if not os.path.exists(save_path):
        os.makedirs(save_path)
 
if args.saved == 1:
    T_arr = np.array([0.0, 0.33, 0.5, 0.7, 0.85])
    vip_hyperparam = np.load(save_path + "/vip_hyperparam.npy")
    sp_hyperparam = np.load(save_path + "/sp_hyperparam.npy")
    vip = vip_hyperparam[np.where(T == T_arr)[0][0]]
    sp = sp_hyperparam[np.where(T == T_arr)[0][0]]


sz=32

model = CNN_switch_3L(sz, int(sp), int(vip), int(vip))

model.double()
if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
error = nn.CrossEntropyLoss()

def train(model, dataT, dataV):
    model.train()

    acc_test_duringL = []
    acc_train_duringL = []

    for batch_idx, (data, target) in enumerate(dataT):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, switch, s_pr)

        predicted = torch.max(output.data, 1)[1]

        loss = error(output, target)

        loss.backward()
        optimizer.step()

        acc_train_duringL.append(float((predicted.to(device) == target).sum())/len(target))

        if batch_idx % 100 == 0:
            acc = validate(model, dataV)
            acc_test_duringL.append(acc)

        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(dataT.dataset),
        #        100. * batch_idx / len(dataT), loss.item()))

    return acc_train_duringL, acc_test_duringL

def validate(model, dataV):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    for data, target in dataV:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data, switch, s_pr)
        # sum up batch loss
        val_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        total += len(target)

    val_loss /= total
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total,
        100. * correct / total))

    return 100. * correct / total

s_pr = 0
switch = 0

acc_train = []
acc_test = []

for name, param in model.named_parameters():
        if ('vip' in name):
            param.requires_grad = False

for name, param in model.named_parameters():
    print(name, param.requires_grad)

for epoch in range(1,epochs+1):
    print(epoch)
    [accTrain, accTest] = train(model, train_loader_D1, test_loader_D1)
    acc_train.append(accTrain); acc_test.append(accTest)

acc_train_D1 =  [item for subl in acc_train for item in subl]
acc_test_D1 =  [item for subl in acc_test for item in subl]

acc_initial_D1 = validate(model, test_loader_D1)
acc_initial_D2 = validate(model, test_loader_D2)
print("accuracy of testing the first dataset before switch is on {}%".format(acc_initial_D1))
print("accuracy of testing the second dataset before switch is on {}%".format(acc_initial_D2))

s_pr = 0
switch = 1

acc_train = []
acc_test = []

for name, param in model.named_parameters():
        if ('cnn' in name) | ('fc' in name) | ('out' in name):
            param.requires_grad = False
        else:
            param.requires_grad = True

for name, param in model.named_parameters():
    print(name, param.requires_grad)

for epoch in range(1,epochs+1):
    print(epoch)
    [accTrain, accTest] = train(model, train_loader_D2, test_loader_D2)
    acc_train.append(accTrain); acc_test.append(accTest)

acc_train_D2 =  [item for subl in acc_train for item in subl]
acc_test_D2 =  [item for subl in acc_test for item in subl]

acc_afterT_D2 = validate(model, test_loader_D2)
print("accuracy of testing the second dataset after switch is on {}%".format(acc_afterT_D2))

save_path1 = save_path + '/initial_D1_' + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed-" + str(seed) + ".npy"
torch.save(acc_initial_D1, save_path1)
save_path2 = save_path + '/initial_D2_' + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed-" + str(seed) + ".npy"
torch.save(acc_initial_D2, save_path2)
save_path3 = save_path +  "/afterT_D2_" + T_value + "-vip-" + str(vip) + "-sp-" + str(sp) + "-seed" + str(seed) + ".npy"
torch.save(acc_afterT_D2, save_path3)

if args.saved == 1:
    torch.save(model.state_dict(), save_path + '/' +
    "model_" + str(args.T) +  '-' + str(vip) + '-' + str(sp) + '-' + str(args.seed) + '.pt')
    np.save(save_path + '/' +
    "results_" + str(args.T) +  '-' + str(vip) + '-' + str(sp) + '-' + str(args.seed) + '.pt', np.array([acc_initial_D1, acc_initial_D2, acc_afterT_D2]))

#np.savez(save_path + "/acc_switch_" + str(args.T) +  '-' + str(vip) + '-' + str(sp) + '-' + str(args.seed) + "_wReLU_forComp.npz", acc_train_D1, acc_test_D1, acc_train_D2, acc_test_D2)


