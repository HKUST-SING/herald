#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../tf_launch_server.py

python3 ${mainpy} --config ${workdir}/../settings/tf_local_s1_w2.json --id 0


# python3 tf_launch_server.py --config ./settings/tf_local_s1_w2.json --id 0
