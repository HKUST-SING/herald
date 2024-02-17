#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

# python -m hetu.launcher ${workdir}/../settings/local_s1.yml -n 1 --sched &
# mpirun --allow-run-as-root -np 4 python ${mainpy} --model wdl_criteo --val --comm Hybrid --cache lfuopt --bound 3 --config ${workdir}/../settings/local_w4.yml
# heturun -s 1 -w 2 python ${mainpy} --model wdl_criteo --val --comm Hybrid --cache lfuopt --bound 3
heturun -s 1 -w 2 python ${mainpy} --model fae_wdl_criteo --val --comm Hybrid --cache lfu --bound 0 --bsp 0 --nepoch 1 --all