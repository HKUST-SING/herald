#!/bin/bash

bash hybrid_wdl_criteo_laia.sh > laia.log 2>&1

pkill python

bash hybrid_wdl_criteo.sh > hetu.log 2>&1