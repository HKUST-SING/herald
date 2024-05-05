# cython: language_level=3
from cython.parallel import prange
from libcpp.unordered_set cimport unordered_set
from libcpp.set cimport set
from libcpp.vector cimport vector
import cython
cimport numpy as np
import numpy as np
import os

np.import_array()

cdef extern from "MiniLRUCache.h" namespace "laia_cache":
    cdef cppclass MiniLRUCache:
        # MiniLRUCache(unsigned int) nogil except +
        # MiniLRUCache(int) nogil except +
        MiniLRUCache() nogil except +
        void set_cap(int) nogil
        bint check(int) nogil
        bint is_full() nogil
        int get(int) nogil
        int insert(int) nogil
        void evict(int) nogil
        void outdate(int) nogil
        vector[int] get_keys() nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_dist(Py_ssize_t batch_id, Py_ssize_t start_idx, int num_total_sample, int[:,:] samples_embs, int num_sample, int num_tab, int num_worker, const vector[MiniLRUCache]& cache_snap, int[:,:] samples_dist, int[:,:,:] samples_emb_dep):
    scores_np = np.zeros((num_sample,num_worker), dtype=np.intc)
    cdef int[:,:] scores = scores_np

    # init
    cdef Py_ssize_t i, j 
    cdef int[:,:] count = np.zeros((num_sample, num_worker+1), dtype=np.intc)
    samples_emb_dep[:,:,:] = -1
    
    # cal scores
    cdef int emb, worker, chunksize, num_threads
    cdef bint find_flag
    num_threads = 8
    chunksize = int(num_sample/num_threads)

    for i in prange(num_sample, nogil=True, schedule='static', chunksize=chunksize, num_threads=num_threads):
        for j in range(num_tab * 1):
            emb = samples_embs[i, j]
            find_flag = False
            for worker in range(num_worker):
                if cache_snap[worker].check(emb):
                    scores[i][worker] += 1
                    samples_emb_dep[i, worker, count[i, worker]] = emb
                    count[i, worker] += 1
                    find_flag = True
            if not find_flag:
                samples_emb_dep[i, num_worker, count[i, num_worker]] = emb
                count[i, num_worker] += 1
    
    # dist samples
    workers_workload = np.zeros(num_worker, dtype=int)
    max_workload = num_sample // num_worker

    cdef int max_score, max_score_worker, score
    for i, sample in enumerate(samples_embs):
        max_score = -1
        max_score_worker = -1

        for j in range(num_worker):
            worker = (j + batch_id) % num_worker
            score = scores[i, worker]
            if workers_workload[worker] != max_workload:
                if max_score < score:
                    max_score = score
                    max_score_worker = worker
        samples_dist[max_score_worker, workers_workload[max_score_worker]] = (i + start_idx) % num_total_sample
        workers_workload[max_score_worker] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def laia_scheduler(np.ndarray samples_embs_np, int epoch_num, int mini_batch_size, int batch_num, int nrank, int rank, int cache_size, queue):
    cdef int num_worker, num_sample, num_tab, batch_size

    cdef int[:,:] samples_embs = samples_embs_np
    num_worker = nrank
    num_sample = samples_embs_np.shape[0]
    num_tab = samples_embs_np.shape[1]
    batch_size = mini_batch_size * num_worker

    cdef vector[MiniLRUCache] cache_snap
    cache_snap.resize(num_worker)
    cdef Py_ssize_t idx
    for idx in range(num_worker):
        cache_snap[idx].set_cap(cache_size)

    samples_dist = np.zeros((num_worker, mini_batch_size), dtype=np.intc)
    samples_emb_dep = np.empty((batch_size, num_worker+1, num_tab), dtype=np.intc)
    cdef Py_ssize_t batch_id, epoch_id = 0, i, j, s
    cdef bint in_flag
    cdef int emb
    cdef Py_ssize_t batch_start_idx
    cdef int[:,:] batch_samples, dist_view = samples_dist
    cdef int[:,:,:] dep_view = samples_emb_dep
    cdef vector[unordered_set[int]] comm_plan
    comm_plan.resize(num_worker)
    cdef vector[set[int]] unique_keys
    unique_keys.resize(num_worker)

    while epoch_id < epoch_num or epoch_num < 0:
        epoch_id += 1
        batch_id = 0
        # for the last epoch, we need one more allocation for cache prefetch
        if epoch_id == epoch_num:
            batch_num += 1
        while batch_id < batch_num:
            batch_start_idx = (batch_id * batch_size) % num_sample
            if batch_start_idx + batch_size <= num_sample:
                batch_samples = samples_embs[batch_start_idx : batch_start_idx + batch_size]
            else:
                batch_samples[:num_sample - batch_start_idx] = samples_embs[batch_start_idx:]
                batch_samples[num_sample - batch_start_idx : batch_size] = samples_embs[:batch_size + batch_start_idx - num_sample]

            
            get_dist(batch_id, batch_start_idx, num_sample, batch_samples, batch_size, num_tab, num_worker, cache_snap, samples_dist, samples_emb_dep)

            # create update plan
            for i in prange(num_worker, nogil=True, schedule='static', chunksize=1, num_threads=num_worker): 
                comm_plan[i].clear()
                for s in range(batch_size):
                    in_flag = False
                    for j in range(mini_batch_size):
                        if (s + batch_start_idx) % num_sample == dist_view[i, j]:
                            in_flag = True
                            break
                    if not in_flag:
                        for j in range(num_tab):
                            emb = dep_view[s, i, j]
                            if emb != -1:
                                comm_plan[i].insert(emb)
                            else:
                                break

            queue.put(list(comm_plan[rank]))

            queue.put(samples_dist[rank].copy())

            # update cache snapshot
            for i in prange(num_worker, nogil=True, schedule='static', chunksize=1, num_threads=num_worker):
                for j in comm_plan[i]:
                    cache_snap[i].outdate(j)
                unique_keys[i].clear()
                for j in range(mini_batch_size):
                    for s in range(num_tab):
                        # cache_snap[i].get(samples_embs[dist_view[i, j], s])
                        unique_keys[i].insert(samples_embs[dist_view[i, j], s])
                for j in unique_keys[i]:
                    cache_snap[i].get(j)

            batch_id += 1

    queue.put(-1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef naive_get_dist(Py_ssize_t start_idx, int[:,:] samples_embs, int num_sample, int num_tab, int num_worker, const vector[MiniLRUCache]& cache_snap, int[:,:] samples_dist, int[:,:,:] samples_emb_dep):
    cdef Py_ssize_t i, j 
    cdef int[:,:] count = np.zeros((num_sample, num_worker+1), dtype=np.intc)
    samples_emb_dep[:,:,:] = -1

    cdef int emb, worker, chunksize, num_threads
    cdef bint find_flag
    num_threads = 8
    chunksize = int(num_sample/num_threads)

    for i in prange(num_sample, nogil=True, schedule='static', chunksize=chunksize, num_threads=num_threads):
        for j in range(num_tab * 1):
            emb = samples_embs[i, j]
            find_flag = False
            for worker in range(num_worker):
                if cache_snap[worker].check(emb):
                    samples_emb_dep[i, worker, count[i, worker]] = emb
                    count[i, worker] += 1
                    find_flag = True
            if not find_flag:
                samples_emb_dep[i, num_worker, count[i, num_worker]] = emb
                count[i, num_worker] += 1

    for i in prange(num_worker, nogil=True, schedule='static', chunksize=1, num_threads=num_worker):
        for j in range(num_sample//num_worker):
            samples_dist[i, j] = start_idx + i * (num_sample//num_worker) + j

@cython.boundscheck(False)
@cython.wraparound(False)
def naive_scheduler(np.ndarray samples_embs_np, int epoch_num, int mini_batch_size, int batch_num, int nrank, int rank, int cache_size, queue):
    cdef int num_worker, num_sample, num_tab, batch_size

    cdef int[:,:] samples_embs = samples_embs_np
    num_worker = nrank
    num_sample = samples_embs_np.shape[0]
    num_tab = samples_embs_np.shape[1]
    batch_size = mini_batch_size * num_worker

    cdef vector[MiniLRUCache] cache_snap
    cache_snap.resize(num_worker)
    cdef Py_ssize_t idx
    for idx in range(num_worker):
        cache_snap[idx].set_cap(cache_size)

    samples_dist = np.zeros((num_worker, mini_batch_size), dtype=np.intc)
    samples_emb_dep = np.empty((batch_size, num_worker+1, num_tab), dtype=np.intc)
    cdef Py_ssize_t batch_id, epoch_id = 0, i, j, s
    cdef bint in_flag
    cdef int emb
    cdef Py_ssize_t batch_start_idx
    cdef int[:,:] batch_samples, dist_view = samples_dist
    cdef int[:,:,:] dep_view = samples_emb_dep
    cdef vector[unordered_set[int]] comm_plan
    comm_plan.resize(num_worker)

    while epoch_id < epoch_num or epoch_num < 0:
        epoch_id += 1
        batch_id = 0
        # for the last epoch, we need one more allocation for cache prefetch
        if epoch_id == epoch_num:
            batch_num += 1
        while batch_id < batch_num:
            batch_start_idx = (batch_id * batch_size) % num_sample
            if batch_start_idx + batch_size <= num_sample:
                batch_samples = samples_embs[batch_start_idx : batch_start_idx + batch_size]
            else:
                batch_samples[:num_sample - batch_start_idx] = samples_embs[batch_start_idx:]
                batch_samples[num_sample - batch_start_idx : batch_size] = samples_embs[:batch_size + batch_start_idx - num_sample]

            
            naive_get_dist(batch_start_idx, batch_samples, batch_size, num_tab, num_worker, cache_snap, samples_dist, samples_emb_dep)

            # create update plan
            for i in prange(num_worker, nogil=True, schedule='static', chunksize=1, num_threads=num_worker): 
                comm_plan[i].clear()
                for s in range(batch_size):
                    in_flag = False
                    for j in range(mini_batch_size):
                        if (s + batch_start_idx) % num_sample == dist_view[i, j]:
                            in_flag = True
                            break
                    if not in_flag:
                        for j in range(num_tab):
                            emb = dep_view[s, i, j]
                            if emb != -1:
                                comm_plan[i].insert(emb)
                            else:
                                break
            
            queue.put(list(comm_plan[rank]))

            queue.put(samples_dist[rank].copy())

            # update cache snapshot
            for i in prange(num_worker, nogil=True, schedule='static', chunksize=1, num_threads=num_worker):
                for j in comm_plan[i]:
                    cache_snap[i].outdate(j)
                for j in range(mini_batch_size):
                    for s in range(num_tab):
                        cache_snap[i].get(samples_embs[dist_view[i, j], s])

            batch_id += 1
    queue.put(-1)