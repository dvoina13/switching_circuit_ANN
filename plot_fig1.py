import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default='./fig1_NNs', metavar='P', help='path to save final model (default: ./pretrained)')

args = parser.parse_args()

T_arr = np.array([0.0, 0.33, 0.5, 0.66, 0.7, 0.75, 0.8, 0.85, 0.9])
seed_arr = range(1,11)
data_order = ["noise-then-cifar", "cifar-then-noise"]

acc_initial_step1_noise = np.zeros((len(T_arr),10))
acc_initial_step1_cifar = np.zeros((len(T_arr),10))
acc_initial_step1prime_noise = np.zeros((len(T_arr),10))
acc_initial_step1prime_cifar = np.zeros((len(T_arr),10))
acc_afterT_step2_noise = np.zeros((len(T_arr),10))
acc_afterT_step2_cifar = np.zeros((len(T_arr),10))
acc_afterT_step3_noise = np.zeros((len(T_arr),10))
acc_afterT_step3_cifar = np.zeros((len(T_arr),10))

for ind_T, T in enumerate(T_arr):
    for ind_s, seed in enumerate(seed_arr):
        for ord in data_order:
            T_value = str(T)
            seed_value = str(seed)

            save_path = '/'.join([args.save_path, ord])

            save_path1 = save_path + '/initial_D1_' + T_value + "-seed" + seed_value + ".npy"
            acc_valid_D1 = torch.load(save_path1)
            save_path2 = save_path + '/initial_D2_' + T_value + "-seed" + seed_value + ".npy"
            acc_valid_D2_initial = torch.load(save_path2)
            save_path3 = save_path +  "/afterT_D2_" + T_value + "-seed" + seed_value + ".npy"
            acc_valid_D2_after = torch.load(save_path3)
            save_path4 = save_path + "/afterT_D1_checkCF_" + T_value + "-seed" + seed_value + ".npy"
            acc_valid_D1_after = torch.load(save_path4)

            if ord == "noise-then-cifar":
                acc_initial_step1_noise[ind_T, ind_s] = acc_valid_D1
                acc_initial_step1prime_cifar[ind_T, ind_s] = acc_valid_D2_initial
                acc_afterT_step2_cifar[ind_T, ind_s] = acc_valid_D2_after
                acc_afterT_step3_noise[ind_T, ind_s] = acc_valid_D1_after
            else:
                acc_initial_step1_cifar[ind_T, ind_s] = acc_valid_D1
                acc_initial_step1prime_noise[ind_T, ind_s] = acc_valid_D2_initial
                acc_afterT_step2_noise[ind_T, ind_s] = acc_valid_D2_after
                acc_afterT_step3_cifar[ind_T, ind_s] = acc_valid_D1_after


acc_initial_step1_noise_mean = np.mean(acc_initial_step1_noise, 1)
acc_initial_step1_noise_err = np.std(acc_initial_step1_noise, 1)/np.sqrt(10)
acc_initial_step1_cifar_mean = np.mean(acc_initial_step1_cifar, 1)
acc_initial_step1_cifar_err = np.std(acc_initial_step1_cifar, 1)/np.sqrt(10)

acc_initial_step1prime_noise_mean = np.mean(acc_initial_step1prime_noise, 1)
acc_initial_step1prime_noise_err = np.std(acc_initial_step1prime_noise, 1)/np.sqrt(10)
acc_initial_step1prime_cifar_mean = np.mean(acc_initial_step1prime_cifar, 1)
acc_initial_step1prime_cifar_err = np.std(acc_initial_step1prime_cifar, 1)/np.sqrt(10)

acc_afterT_step2_noise_mean = np.mean(acc_afterT_step2_noise, 1)
acc_afterT_step2_noise_err = np.std(acc_afterT_step2_noise, 1)/np.sqrt(10)
acc_afterT_step2_cifar_mean = np.mean(acc_afterT_step2_cifar, 1)
acc_afterT_step2_cifar_err = np.std(acc_afterT_step2_cifar, 1)/np.sqrt(10)

acc_afterT_step3_noise_mean = np.mean(acc_afterT_step3_noise, 1)
acc_afterT_step3_noise_err = np.std(acc_afterT_step3_noise, 1)/np.sqrt(10)
acc_afterT_step3_cifar_mean = np.mean(acc_afterT_step3_cifar, 1)
acc_afterT_step3_cifar_err = np.std(acc_afterT_step3_cifar, 1)/np.sqrt(10)

plt.figure()
plt.plot(T_arr, acc_initial_step1_noise_mean)
plt.fill_between(T_arr, acc_initial_step1_noise_mean-acc_initial_step1_noise_err, acc_initial_step1_noise_mean+acc_initial_step1_noise_err)
plt.plot(T_arr, acc_initial_step1prime_noise_mean)
plt.fill_between(T_arr, acc_initial_step1prime_noise_mean-acc_initial_step1prime_noise_err, acc_initial_step1prime_noise_mean+acc_initial_step1prime_noise_err)
plt.plot(T_arr, acc_initial_step1prime_noise_mean)
plt.fill_between(T_arr, acc_initial_step1prime_noise_mean-acc_initial_step1prime_noise_err, acc_initial_step1prime_noise_mean+acc_initial_step1prime_noise_err)

plt.figure()
plt.plot(T_arr, acc_initial_step1_cifar_mean)
plt.fill_between(T_arr, acc_initial_step1_cifar_mean-acc_initial_step1_cifar_err, acc_initial_step1_cifar_mean+acc_initial_step1_cifar_err)
plt.plot(T_arr, acc_initial_step1prime_cifar_mean)
plt.fill_between(T_arr, acc_initial_step1prime_cifar_mean-acc_initial_step1prime_cifar_err, acc_initial_step1prime_cifar_mean+acc_initial_step1prime_cifar_err)
plt.plot(T_arr, acc_initial_step1prime_cifar_mean)
plt.fill_between(T_arr, acc_initial_step1prime_cifar_mean-acc_initial_step1prime_cifar_err, acc_initial_step1prime_cifar_mean+acc_initial_step1prime_cifar_err)
