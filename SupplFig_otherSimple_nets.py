import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import argparse
import os

from data_loader import load_training_testing_data
from model_suppl_otherSimple_nets import CNN_simple_switch_const, CNN_wInput, CNN_wOutput


parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, default=0.0, required=False, help="Set Transparency")
parser.add_argument("--T2_use", type=int, default=0, required=False, help="Binary value showing is we are using 2nd transparency")
parser.add_argument("--T2", type=float, default=0.5, required=False, help="Set 2nd value for Transparency (for second dataset)")
parser.add_argument("--seed", type=int, default=1, required=False, help="seed")

parser.add_argument("--data_order", type=str, default="cifar-first", required=False, help="MNIST+noise of MNIST+cifar first?")
parser.add_argument("--network", type=str, default="context input", required=False, help="CHOOSE between options which network to test: ..."
        "context input for having a contextual input, context output for having a contextual output, simple switch, for having a simple switching network")

parser.add_argument("--batch_size", type=int, default=50, required=False, help="batch size for training")
parser.add_argument("--test_batch_size", type=int, default=1000, required=False, help="batch size for training")
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
parser.add_argument('--data-dir', type=str, default='/home/dvoina/CNNEx_project2/', metavar='P', help='path to data')
parser.add_argument('--save-path', type=str, default='./SupplFig_simple_nets', metavar='P', help='path to save final model')

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

network = args.network

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
    #ord = "noise-then-cifar_wInput"
    ord = "noise-then-cifar_wOutput"
    #ord = "noise-then-cifar_wSimpleSwitch"
else:
    train_loader_D1 = train_loader_cifar
    test_loader_D1 = test_loader_cifar
    train_loader_D2 = train_loader_noise
    test_loader_D2 = test_loader_noise
    #ord = "cifar-then-noise_wInput"
    ord = "cifar-then-noise_wOutput"
    #ord = "cifar-then-noise_wSimpleSwitch"

sz=32

if network == "context-input":
    model = CNN_wInput(sz)
elif network == "context-output":
    model = CNN_wOutput(sz)
elif network == "simple-switch":
    model = CNN_simple_switch_const(sz,100)

model.double()
if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
error = nn.CrossEntropyLoss()
error2 = nn.MSELoss()

def train(model, dataT, params, network):
    model.train()

    for batch_idx, (data, target) in enumerate(dataT):
        if cuda:
            data, target = data.cuda().double(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        if network == "context-input":

            c = params["context"]
            output = model(data, c)
            loss = error(output, target)
            loss.backward()

        elif network == "context-output":

            context = params["context"]
            if context == 0:
                target_context = torch.zeros(target.size()[0])
            else:
                target_context = torch.ones(target.size()[0])
                
            if cuda:
                target_context = target_context.cuda()

            output = model(data)
            loss = error(output[:, 0:10], target)
            loss.backward(retain_graph = True)
            loss2 = error2(output[:, 10].float(), target_context)
            loss2.backward()

        elif network == "simple-switch":

            sw = params["switch"]
            s_pr = params["sum_or_product"]
            device = params["device"]
            output = model(data, sw, s_pr, device)
            loss = error(output, target)
            loss.backward()

        optimizer.step()

def validate(model, dataV, params, network):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    for data, target in dataV:
        if cuda:
            data, target = data.cuda().double(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        if network == "context-input":

            c = params["context"]
            output = model(data, c)
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]

        elif network == "context-output":

            output = model(data)
            val_loss += F.nll_loss(output[:,0:10], target, size_average=False).item()
            pred = output[:, 0:10].data.max(1, keepdim=True)[1]

        elif network == "simple-switch":

            sw = params["switch"]
            s_pr = params["sum_or_product"]
            device = params["device"]
            output = model(data, sw, s_pr, device)
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]


        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        total += len(target)

    val_loss /= total
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, total, 100. * correct / total))

    return 100. * correct / total

params = {}
if network == "simple-switch":
    params["switch"] = 0
    params["sum_or_product"] = 0
    params["device"] = device
    
    for name, param in model.named_parameters():
        if "vip" in name:
            param.requires_grad = False
            
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
            
else:
    params["context"] = 0

for epoch in range(1,epochs+1):
    print(epoch)
    train(model, train_loader_D1, params, network)
    validate(model, test_loader_D1, params, network)

acc_valid_D1 = validate(model, test_loader_D1, params, network)
print('Test accuracy for dataset 1: {}%'.format(acc_valid_D1))

if network != "simple-switch":
    params["context"] = 1
    
acc_valid_D2_initial = validate(model, test_loader_D2, params, network)
print('Test accuracy for dataset 1: {}%'.format(acc_valid_D2_initial))

if network == "simple-switch":
    params["switch"] = 1
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if "vip" in name:
            param.requires_grad = True
        
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
else:
    params["context"] = 1

for epoch in range(1,epochs+1):
    print(epoch)
    train(model, train_loader_D2, params, network)
    validate(model, test_loader_D2, params, network)

acc_valid_D2_after = validate(model, test_loader_D2, params, network)
print('Test accuracy for dataset 2 (after training): {}%'.format(acc_valid_D2_after))

if network != "simple-switch":
    
    params["context"] = 0
    acc_valid_D1_after_CF = validate(model, test_loader_D1, params, network)
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
if network != "simple-switch":
    save_path4 = save_path + "/afterT_D1_checkCF_" + T_value + "-seed" + str(seed) + ".npy"
    torch.save(acc_valid_D1_after_CF, save_path4)

