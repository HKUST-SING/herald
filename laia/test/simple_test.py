import os
import sys
import multiprocessing as mp
import numpy as np
from time import sleep

# c++
sys.path.append("../build/")
import laia_cache

def process_partial_criteo_data(path="/herald/data/criteo/"):
    file_path = os.path.join(path, "train_sparse_feats.npy")
    if not os.path.exists(file_path):
        assert False, "No dataset files in {}".format(path)
    return np.load(file_path, mmap_mode='r')

def test():
    sparse = process_partial_criteo_data()
    sparse = sparse.astype(np.intc)
    batch_size = int(128)
    nrank = int(8)
    rank = int(0)
    batch_num = len(sparse)//nrank//batch_size
    cache_limit = int(5000)
    epoch_num = int(1)
    thread_num = 16
    top_k_table = 5

    # c++
    cpp_scheduler = laia_cache.LaiaScheduler()
    cpp_scheduler.start(sparse, sparse.shape[0], sparse.shape[1], epoch_num, batch_size, batch_num, nrank, rank, cache_limit, thread_num, top_k_table)

    sleep(1)

    for i in range(batch_num):
        cpp_comm_plan = cpp_scheduler.pop()
        cpp_dist = cpp_scheduler.pop()
    
    cpp_scheduler.report_cache_perf("wdl_criteo", "cache_perf.txt")

if __name__ == "__main__":
    test()