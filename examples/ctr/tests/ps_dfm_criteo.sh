#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

# python ${mainpy} --model dfm_criteo --val --comm PS --cache lfuopt --bound 3 --config ${workdir}/../settings/local_s1_w4.yml
heturun -s 1 -w 2 python ${mainpy} --model dfm_criteo --val --comm PS --cache lfuopt --bound 3