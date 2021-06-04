#!/bin/bash

#  run_experiment.sh
#  
#
#  Created by Doris V on 5/3/21.
#

source activate pytorch

seed_arr=($(seq 1 1 10))
#T_arr=(0.0 0.33 0.5 0.66 0.7 0.75 0.8 0.85 0.9 0.95)
T_arr=(0.0 0.33 0.5 0.7 0.85 0.66 0.75 0.8 0.9 0.95)
data_order=("noise-first" "cifar-first")
sp_arr=(3)
vip_arr=(10)

#for seed in "${seed_arr[@]}"
#do
#    for T in "${T_arr[@]}"
#    do
#        for d_ord in "${data_order[@]}"
#        do
#            python train_fig1.py --T=$T --seed=$seed --data_order=$d_ord
#        done
#    done
#done

#python plot_fig1.py

for seed in "${seed_arr[@]}"
do
    for T in "${T_arr[@]}"
    do
        for d_ord in "${data_order[@]}"
        do
            for sp in "${sp_arr[@]}"
            do
                for vip in "${vip_arr[@]}"
                do
                    python train_fig3.py --T=$T --seed=$seed --data_order=$d_ord --sp=$sp --vip=$vip
                done
            done
        done
    done
done

#python find_hyperparam_sp_vip_fig3.py --data_order2=0
#python find_hyperparam_sp_vip_fig3.py --data_order2=1

#for seed in "${seed_arr[@]}"
#do
#    for T in "${T_arr[@]}"
#    do
#        for d_ord in "${data_order[@]}"
#        do

#            python train_fig3.py --T=$T --seed=$seed --data_order=$d_ord --saved=1
            
#        done
#    done
#done

#python plot_fig3.py


#for seed in "${seed_arr[@]}"
#do
#    for T in "${T_arr[@]}"
#    do
#        for d_ord in "${data_order[@]}"
#        do
#            python train_fig5.py --T=$T --seed=$seed --data_order=$d_ord
#        done
#    done
#done
