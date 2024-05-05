#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

# python ${mainpy} --model dcn_criteo --val
heturun -w 1 python ${mainpy} --model dcn_criteo --cache lru --bound 0 --bsp 0 --nepoch 1 --all