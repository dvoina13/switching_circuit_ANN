#!/bin/bash

#  run_suppl.sh
#  
#
#  Created by Doris V on 5/18/21.
#  

source activate pytorch

seed_arr=($(seq 1 1 10))
#T_arr=(0.0 0.33 0.5 0.66 0.7 0.75 0.8 0.85 0.9 0.95)
T_arr=(0.0 0.33 0.5 0.7 0.85 0.66 0.75 0.8 0.9 0.95)
data_order=("noise-first" "cifar-first")
sp_arr=(3)
vip_arr=(10)
net=("context-output")

for seed in "${seed_arr[@]}"
do
    for T in "${T_arr[@]}"
    do
        for d_ord in "${data_order[@]}"
        do
            for n in "${net[@]}"
            do
                python SupplFig_otherSimple_nets.py --T=$T --seed=$seed --data_order=$d_ord --network=$n
            done
        done
    done
done
