import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def load_training_testing_data(T_value, T2_use, T2_value, batch_size, test_batch_size, data_dir="", num_workers=4, pin_memory=False):
    
    x_train_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_noise_blur" + T_value + ".npy")
    x_train_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_cifar_blur" + T_value +".npy")
    x_test_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_noise_blur" + T_value + ".npy")
    x_test_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_cifar_blur" + T_value + ".npy")
    targetsTrain_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_cifar_blur" + T_value + ".npy")
    targetsTrain_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_noise_blur" + T_value + ".npy")
    targetsTest_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_cifar_blur" + T_value + ".npy")
    targetsTest_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_noise_blur" + T_value + ".npy")

    #arr_digitID = np.load("datasets_mnist_cifar_noise4/arrID_blur" + blur_value + ".npy")
    #targets_cifar_permuted = np.load('datasets_mnist_cifar_noise4/targets_cifar_permute' + blur_value + ".npy")
    #targets_noise_permuted = np.load('datasets_mnist_cifar_noise4/targets_noise_permute' + blur_value + ".npy")

    x_train_cifar = torch.from_numpy(x_train_cifar).unsqueeze(1)
    x_train_noise = torch.from_numpy(x_train_noise).unsqueeze(1)
    x_test_cifar = torch.from_numpy(x_test_cifar).unsqueeze(1)
    x_test_noise = torch.from_numpy(x_test_noise).unsqueeze(1)

    targets_train_cifar = torch.from_numpy(targetsTrain_cifar).type(torch.LongTensor).squeeze()
    targets_train_noise = torch.from_numpy(targetsTrain_noise).type(torch.LongTensor).squeeze()
    targets_test_cifar = torch.from_numpy(targetsTest_cifar).type(torch.LongTensor).squeeze()
    targets_test_noise = torch.from_numpy(targetsTest_noise).type(torch.LongTensor).squeeze()

    train_cifar = torch.utils.data.TensorDataset(x_train_cifar, targets_train_cifar)
    train_noise = torch.utils.data.TensorDataset(x_train_noise, targets_train_noise)
    test_cifar = torch.utils.data.TensorDataset(x_test_cifar, targets_test_cifar)
    test_noise = torch.utils.data.TensorDataset(x_test_noise, targets_test_noise)

    train_loader_cifar = DataLoader(train_cifar, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader_cifar = DataLoader(test_cifar, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_noise = DataLoader(train_noise, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader_noise = DataLoader(test_noise, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if T2_use:
        x_train_noise2 = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_noise_blur" + T2_value + ".npy")
        x_train_cifar2 = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_cifar_blur" + T2_value + ".npy")
        x_test_noise2 = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_noise_blur" + T2_value + ".npy")
        x_test_cifar2 = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_cifar_blur" + T2_value + ".npy")
        targetsTrain_cifar2 = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_cifar_blur" + T2_value + ".npy")
        targetsTrain_noise2 = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_noise_blur" + T2_value + ".npy")
        targetsTest_cifar2 = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_cifar_blur" + T2_value + ".npy")
        targetsTest_noise2 = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_noise_blur" + T2_value + ".npy")

        x_train_cifar2 = torch.from_numpy(x_train_cifar2).unsqueeze(1)
        x_train_noise2 = torch.from_numpy(x_train_noise2).unsqueeze(1)
        x_test_cifar2 = torch.from_numpy(x_test_cifar2).unsqueeze(1)
        x_test_noise2 = torch.from_numpy(x_test_noise2).unsqueeze(1)

        targets_train_cifar2 = torch.from_numpy(targetsTrain_cifar2).type(torch.LongTensor).squeeze()
        targets_train_noise2 = torch.from_numpy(targetsTrain_noise2).type(torch.LongTensor).squeeze()
        targets_test_cifar2 = torch.from_numpy(targetsTest_cifar2).type(torch.LongTensor).squeeze()
        targets_test_noise2 = torch.from_numpy(targetsTest_noise2).type(torch.LongTensor).squeeze()

        train_cifar2 = torch.utils.data.TensorDataset(x_train_cifar2, targets_train_cifar2)
        train_noise2 = torch.utils.data.TensorDataset(x_train_noise2, targets_train_noise2)
        test_cifar2 = torch.utils.data.TensorDataset(x_test_cifar2,targets_test_cifar2)
        test_noise2 = torch.utils.data.TensorDataset(x_test_noise2,targets_test_noise2)

        train_loader_cifar2 = DataLoader(train_cifar2, batch_size = batch_size, shuffle = False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader_cifar2 = DataLoader(test_cifar2, batch_size = test_batch_size, shuffle = False, num_workers=num_workers, pin_memory=pin_memory)
        train_loader_noise2 = DataLoader(train_noise2, batch_size = batch_size, shuffle = False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader_noise2 = DataLoader(test_noise2, batch_size = test_batch_size, shuffle = False, num_workers=num_workers, pin_memory=pin_memory)

        return (train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar, train_loader_noise2, test_loader_noise2, train_loader_cifar2, test_loader_cifar2)

    else:

        return (train_loader_noise, test_loader_noise, train_loader_cifar, test_loader_cifar)


def load_training_validation_testing_data(T_value, batch_size, test_batch_size, seed, data_dir="", shuffle = False, valid_size = 0.2, num_workers=4, pin_memory=False):

    x_train_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_noise_blur" + T_value + ".npy")
    x_train_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_cifar_blur" + T_value +".npy")
    x_test_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_noise_blur" + T_value + ".npy")
    x_test_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_cifar_blur" + T_value + ".npy")
    targetsTrain_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_cifar_blur" + T_value + ".npy")
    targetsTrain_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_noise_blur" + T_value + ".npy")
    targetsTest_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_cifar_blur" + T_value + ".npy")
    targetsTest_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_noise_blur" + T_value + ".npy")

    #arr_digitID = np.load("datasets_mnist_cifar_noise4/arrID_blur" + blur_value + ".npy")
    #targets_cifar_permuted = np.load('datasets_mnist_cifar_noise4/targets_cifar_permute' + blur_value + ".npy")
    #targets_noise_permuted = np.load('datasets_mnist_cifar_noise4/targets_noise_permute' + blur_value + ".npy")

    x_train_cifar = torch.from_numpy(x_train_cifar).unsqueeze(1)
    x_train_noise = torch.from_numpy(x_train_noise).unsqueeze(1)
    x_test_cifar = torch.from_numpy(x_test_cifar).unsqueeze(1)
    x_test_noise = torch.from_numpy(x_test_noise).unsqueeze(1)

    targets_train_cifar = torch.from_numpy(targetsTrain_cifar).type(torch.LongTensor).squeeze()
    targets_train_noise = torch.from_numpy(targetsTrain_noise).type(torch.LongTensor).squeeze()
    targets_test_cifar = torch.from_numpy(targetsTest_cifar).type(torch.LongTensor).squeeze()
    targets_test_noise = torch.from_numpy(targetsTest_noise).type(torch.LongTensor).squeeze()

    train_cifar = torch.utils.data.TensorDataset(x_train_cifar, targets_train_cifar)
    train_noise = torch.utils.data.TensorDataset(x_train_noise, targets_train_noise)
    test_cifar = torch.utils.data.TensorDataset(x_test_cifar, targets_test_cifar)
    test_noise = torch.utils.data.TensorDataset(x_test_noise, targets_test_noise)
    
    num_train = len(train_cifar)
    indices = list(range(num_train))
    split = int(np.floor((1 - valid_size) * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader_cifar = torch.utils.data.DataLoader(
        train_cifar, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    train_loader_noise = torch.utils.data.DataLoader(
        train_noise, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader_cifar = torch.utils.data.DataLoader(
        train_cifar, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader_noise = torch.utils.data.DataLoader(
        train_noise, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader_cifar = torch.utils.data.DataLoader(
        test_cifar, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader_noise = torch.utils.data.DataLoader(
        test_noise, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader_noise, valid_loader_noise, test_loader_noise, train_loader_cifar, valid_loader_cifar, test_loader_cifar)
    
    
def load_training_testing_data_permuted(T_value, batch_size, test_batch_size, data_dir="", num_workers=4, pin_memory=False):

    x_train_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_noise_blur" + T_value + ".npy")
    x_train_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/x_train_cifar_blur" + T_value +".npy")
    x_test_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_noise_blur" + T_value + ".npy")
    x_test_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/x_test_cifar_blur" + T_value + ".npy")
    targetsTrain_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_cifar_blur" + T_value + ".npy")
    targetsTrain_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_train_noise_blur" + T_value + ".npy")
    targetsTest_cifar = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_cifar_blur" + T_value + ".npy")
    targetsTest_noise = np.load(data_dir + "datasets_mnist_cifar_noise4/targets_test_noise_blur" + T_value + ".npy")

    #arr_digitID = np.load("datasets_mnist_cifar_noise4/arrID_blur" + T_value + ".npy")
    targets_cifar_permuted = np.load(data_dir+'datasets_mnist_cifar_noise4/targets_cifar_permute' + T_value + ".npy")
    targets_noise_permuted = np.load(data_dir+'datasets_mnist_cifar_noise4/targets_noise_permute' + T_value + ".npy")

    targets_train_cifar_permuted = targets_cifar_permuted[0:np.shape(x_train_cifar)[0]]
    targets_test_cifar_permuted = targets_cifar_permuted[np.shape(x_train_cifar)[0]:]
    targets_train_noise_permuted = targets_noise_permuted[0:np.shape(x_train_noise)[0]]
    targets_test_noise_permuted = targets_noise_permuted[np.shape(x_train_noise)[0]:]
    
    x_train_cifar = torch.from_numpy(x_train_cifar).unsqueeze(1)
    x_train_noise = torch.from_numpy(x_train_noise).unsqueeze(1)
    x_test_cifar = torch.from_numpy(x_test_cifar).unsqueeze(1)
    x_test_noise = torch.from_numpy(x_test_noise).unsqueeze(1)

    targets_train_cifar = torch.from_numpy(targetsTrain_cifar).type(torch.LongTensor).squeeze()
    targets_train_noise = torch.from_numpy(targetsTrain_noise).type(torch.LongTensor).squeeze()
    targets_test_cifar = torch.from_numpy(targetsTest_cifar).type(torch.LongTensor).squeeze()
    targets_test_noise = torch.from_numpy(targetsTest_noise).type(torch.LongTensor).squeeze()

    targets_train_cifar_permuted = torch.from_numpy(targets_train_cifar_permuted).type(torch.LongTensor).squeeze()
    targets_train_noise_permuted = torch.from_numpy(targets_train_noise_permuted).type(torch.LongTensor).squeeze()
    targets_test_cifar_permuted = torch.from_numpy(targets_test_cifar_permuted).type(torch.LongTensor).squeeze()
    targets_test_noise_permuted = torch.from_numpy(targets_test_noise_permuted).type(torch.LongTensor).squeeze()

    
    train_cifar_permuted = torch.utils.data.TensorDataset(x_train_cifar, targets_train_cifar)
    train_noise_permuted = torch.utils.data.TensorDataset(x_train_noise, targets_train_noise_permuted)
    test_cifar_permuted = torch.utils.data.TensorDataset(x_test_cifar,targets_test_cifar)
    test_noise_permuted = torch.utils.data.TensorDataset(x_test_noise,targets_test_noise_permuted)

    train_loader_cifar_permuted = DataLoader(train_cifar_permuted, batch_size = batch_size, shuffle = False)
    test_loader_cifar_permuted = DataLoader(test_cifar_permuted, batch_size = batch_size, shuffle = False)
    train_loader_noise_permuted = DataLoader(train_noise_permuted, batch_size = batch_size, shuffle = False)
    test_loader_noise_permuted = DataLoader(test_noise_permuted, batch_size = batch_size, shuffle = False)

    return (train_loader_noise_permuted, test_loader_noise_permuted, train_loader_cifar_permuted, test_loader_cifar_permuted)



