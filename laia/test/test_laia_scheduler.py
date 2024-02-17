import os
import sys
import numpy as np
import time

sys.path.append("../build/")
import laia_cache

num_sample = 10000
num_table = 27
epoch_num = 1
mini_batch_size = 128
batch_num = num_sample // mini_batch_size
nrank = 8
rank = 0
cache_size = int(0.1 * num_sample * 8)
num_thread = 16
top_k_table = 5

sample_keys = np.random.randint(200000, size=(num_sample, num_table))
scheduler = laia_cache.LaiaScheduler()
scheduler.start(
    sample_keys,
    num_sample,
    num_table,
    epoch_num,
    mini_batch_size,
    batch_num,
    nrank,
    rank,
    cache_size,
    num_thread,
    top_k_table,
)

# dist = scheduler.pop()
i = 0
# time.sleep(20)
while i < batch_num:
    res = scheduler.pop()
    print("{} : len {}".format(type(res), len(res)))
    # print(res)
    res = scheduler.pop()
    print("{} : len {}".format(type(res), len(res)))
    i = i + 1
