import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default='./fig3_NNs', metavar='P', help='path to save final model (default: ./pretrained)')

args = parser.parse_args()

save_path = args.save_path

data_order = ["noise-then-cifar", "cifar-then-noise"]
T_arr = np.array([0.0, 0.33, 0.5, 0.7, 0.85])
seed_arr = range(1,11)

for do_ind, do in enumerate(data_order):
    s_path = '/'.join([save_path, do])

    vip_hyperparam = np.load(s_path + "/vip_hyperparam.npy")
    sp_hyperparam = np.load(s_path + "/sp_hyperparam.npy")

    ALLacc_initial_D1 = np.load(s_path + "/acc_initial_D1_mean.npy")
    ALLacc_initial_D2 = np.load(s_path + "/acc_initial_D2_mean.npy")
    ALLacc_afterT_D2 = np.load(s_path + "/acc_afterT_D2_mean.npy")

    ALLacc_initial_D1 = np.mean(ALLacc_initial_D1, 3)
    ALLacc_initial_D2 = np.mean(ALLacc_initial_D2, 3)
    ALLacc_afterT_D2 = np.mean(ALLacc_afterT_D2, 3)

    plt.figure()
    plt.imshow(ALLacc_afterT_D2[0, :, :])
    plt.colorbar()
    plt.title("heat map of accuracies T=0.0")

    plt.figure()
    plt.imshow(ALLacc_afterT_D2[1, :, :])
    plt.colorbar()
    plt.title("heat map of accuracies T=0.33")

    plt.figure()
    plt.imshow(ALLacc_afterT_D2[2, :, :])
    plt.colorbar()
    plt.title("heat map of accuracies given T=0.5")

    plt.figure()
    plt.imshow(ALLacc_afterT_D2[3, :, :])
    plt.colorbar()
    plt.title("heat map of accuracies given T=0.7")

    plt.figure()
    plt.imshow(ALLacc_afterT_D2[4, :, :])
    plt.colorbar()
    plt.title("heat map of accuracies given T=0.85")



vip_arr = np.array([1, 5, 10, 20])
sp_arr = np.array([1, 3, 5])


def plot_T_vs_acc(best_params, vip=5, sp=5):
    accuracy_initial_D1 = np.zeros((len(data_order), len(T_arr), 10))
    accuracy_initial_D2 = np.zeros((len(data_order), len(T_arr), 10))
    accuracy_afterT_D2 = np.zeros((len(data_order), len(T_arr), 10))

    for do_ind, do in enumerate(data_order):

        s_path = '/'.join([save_path, do])

        vip_hyperparam = np.load(s_path + "/vip_hyperparam.npy")
        sp_hyperparam = np.load(s_path + "/sp_hyperparam.npy")

        for T_ind, T in enumerate(T_arr):
            for seed in range(1, 11):

                if best_params == 1:
                    vip = vip_hyperparam[np.where(T == T_arr)[0][0]]
                    sp = sp_hyperparam[np.where(T == T_arr)[0][0]]

                    # model_param = torch.load(save_path + '/' + "model_" + str(args.T) + '-' + str(vip) + '-' + str(sp) + '-' + str(args.seed) + '.pt')
                    [acc_initial_D1, acc_initial_D2, acc_afterT_D2] = np.load(
                        s_path + '/' + "results_" + str(T) + '-' + str(vip) + '-' + str(sp) + '-' + str(
                            seed) + '.pt.npy')

                else:
                    acc_initial_D1 = torch.load(
                        s_path + '/initial_D1_' + str(T) + "-vip-" + str(int(vip)) + "-sp-" + str(
                            int(sp)) + "-seed-" + str(seed) + ".npy")
                    acc_initial_D2 = torch.load(
                        s_path + '/initial_D2_' + str(T) + "-vip-" + str(int(vip)) + "-sp-" + str(
                            int(sp)) + "-seed-" + str(seed) + ".npy")
                    acc_afterT_D2 = torch.load(s_path + '/afterT_D2_' + str(T) + "-vip-" + str(int(vip)) + "-sp-" + str(
                        int(sp)) + "-seed" + str(seed) + ".npy")

                accuracy_initial_D1[do_ind, T_ind, seed - 1] = acc_initial_D1;
                accuracy_initial_D2[do_ind, T_ind, seed - 1] = acc_initial_D2;
                accuracy_afterT_D2[do_ind, T_ind, seed - 1] = acc_afterT_D2;

    err_initial_D1 = np.std(accuracy_initial_D1, 2) / np.sqrt(10)
    err_initial_D2 = np.std(accuracy_initial_D2, 2) / np.sqrt(10)
    err_afterT_D2 = np.std(accuracy_afterT_D2, 2) / np.sqrt(10)

    accuracy_initial_D1 = np.mean(accuracy_initial_D1, 2)
    accuracy_initial_D2 = np.mean(accuracy_initial_D2, 2)
    accuracy_afterT_D2 = np.mean(accuracy_afterT_D2, 2)

    plt.figure()
    plt.plot(T_arr, accuracy_initial_D1[0, :], "-*")
    plt.fill_between(T_arr, accuracy_initial_D1[0, :] - err_initial_D1[0, :],
                     accuracy_initial_D1[0, :] + err_initial_D1[0, :], alpha=0.5, facecolor="lightblue")
    plt.plot(T_arr, accuracy_initial_D2[1, :], "-*")
    plt.fill_between(T_arr, accuracy_initial_D2[1, :] - err_initial_D2[1, :],
                     accuracy_initial_D2[1, :] + err_initial_D2[1, :], alpha=0.5, facecolor="moccasin")
    plt.plot(T_arr, accuracy_afterT_D2[1, :], "-*")
    plt.fill_between(T_arr, accuracy_afterT_D2[1, :] - err_afterT_D2[1, :],
                     accuracy_afterT_D2[1, :] + err_afterT_D2[1, :], alpha=0.5, facecolor="lightgreen")

    plt.figure()
    plt.plot(T_arr, accuracy_initial_D1[1, :], "-*")
    plt.fill_between(T_arr, accuracy_initial_D1[1, :] - err_initial_D1[1, :],
                     accuracy_initial_D1[1, :] + err_initial_D1[1, :], alpha=0.5, facecolor="lightblue")
    plt.plot(T_arr, accuracy_initial_D2[0, :], "-*")
    plt.fill_between(T_arr, accuracy_initial_D2[0, :] - err_initial_D2[0, :],
                     accuracy_initial_D2[0, :] + err_initial_D2[0, :], alpha=0.5, facecolor="moccasin")
    plt.plot(T_arr, accuracy_afterT_D2[0, :], "-*")
    plt.fill_between(T_arr, accuracy_afterT_D2[0, :] - err_afterT_D2[0, :],
                     accuracy_afterT_D2[0, :] + err_afterT_D2[0, :], alpha=0.5, facecolor="lightgreen")


plot_T_vs_acc(1)
plot_T_vs_acc(0,1,5)
plot_T_vs_acc(0,5,5)
plot_T_vs_acc(0,5,3)
plot_T_vs_acc(0,10,5)


def plot_vip_vs_acc(T, sp):
    accuracy_initial_D1 = np.zeros((len(data_order), len(vip_arr), 10))
    accuracy_initial_D2 = np.zeros((len(data_order), len(vip_arr), 10))
    accuracy_afterT_D2 = np.zeros((len(data_order), len(vip_arr), 10))

    acc_allVIPs_afterT_D2 = np.zeros((len(data_order), len(vip_arr) + 1, 10))

    for do_ind, do in enumerate(data_order):

        s_path = '/'.join([save_path, do])

        for seed in range(1, 11):
            for vip_ind, vip in enumerate(vip_arr):
                acc_initial_D1 = torch.load(s_path + '/initial_D1_' + str(T) + "-vip-" + str(int(vip)) + "-sp-" + str(
                    int(sp)) + "-seed-" + str(seed) + ".npy")
                acc_initial_D2 = torch.load(s_path + '/initial_D2_' + str(T) + "-vip-" + str(int(vip)) + "-sp-" + str(
                    int(sp)) + "-seed-" + str(seed) + ".npy")
                acc_afterT_D2 = torch.load(s_path + '/afterT_D2_' + str(T) + "-vip-" + str(int(vip)) + "-sp-" + str(
                    int(sp)) + "-seed" + str(seed) + ".npy")

                accuracy_initial_D1[do_ind, vip_ind, seed - 1] = acc_initial_D1;
                accuracy_initial_D2[do_ind, vip_ind, seed - 1] = acc_initial_D2;
                accuracy_afterT_D2[do_ind, vip_ind, seed - 1] = acc_afterT_D2;

    vip_arr2 = np.concatenate((np.array([0]), np.array(vip_arr)))
    acc_allVIPs_afterT_D2[1, 1:, :] = accuracy_afterT_D2[1, :, :];
    acc_allVIPs_afterT_D2[1, 0, :] = np.mean(accuracy_initial_D2[1, :, :], 0);
    acc_allVIPs_afterT_D2[0, 1:, :] = accuracy_afterT_D2[0, :, :];
    acc_allVIPs_afterT_D2[0, 0, :] = np.mean(accuracy_initial_D2[0, :, :], 0);

    err_allVIPs_afterT_D2 = np.std(acc_allVIPs_afterT_D2, 2) / np.sqrt(10)
    accuracy_allVIPs_afterT_D2 = np.mean(acc_allVIPs_afterT_D2, 2)

    plt.figure()
    plt.plot(vip_arr2, accuracy_allVIPs_afterT_D2[1, :], "-*", color="green")
    plt.fill_between(vip_arr2, accuracy_allVIPs_afterT_D2[1, :] - err_allVIPs_afterT_D2[1, :],
                     accuracy_allVIPs_afterT_D2[1, :] + err_allVIPs_afterT_D2[1, :], alpha=0.5, facecolor="lightgreen")

    plt.figure()
    plt.plot(vip_arr2, accuracy_allVIPs_afterT_D2[0, :], "-*", color="green")
    plt.fill_between(vip_arr2, accuracy_allVIPs_afterT_D2[0, :] - err_allVIPs_afterT_D2[0, :],
                     accuracy_allVIPs_afterT_D2[0, :] + err_allVIPs_afterT_D2[0, :], alpha=0.5, facecolor="lightgreen")


plot_vip_vs_acc(0.0, 3)
plot_vip_vs_acc(0.33, 3)
plot_vip_vs_acc(0.5, 3)
plot_vip_vs_acc(0.85, 3)

plot_vip_vs_acc(0.0, 5)
plot_vip_vs_acc(0.33, 5)
plot_vip_vs_acc(0.5, 5)
plot_vip_vs_acc(0.85, 5)