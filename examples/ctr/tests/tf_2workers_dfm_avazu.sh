#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../tf_launch_worker.py

rm -f logs/temp*.log
CUDA_VISIBLE_DEVICES=0 python3 ${mainpy} --model dfm_avazu --config ${workdir}/../settings/tf_local_s1_w2.json --rank 0 --all > ${workdir}/../logs/temp0_avazu.log & 
CUDA_VISIBLE_DEVICES=1 python3 ${mainpy} --model dfm_avazu --config ${workdir}/../settings/tf_local_s1_w2.json --rank 1 --all > ${workdir}/../logs/temp1_avazu.log & 
wait
