# Herald

Herald is a high-performance distributed deep learning training system targeting for embedding models. It optimizes the embedding transmissions between workers and parameter servers with embedding scheduling.

Herald is built on top of [Hetu](https://github.com/Hsword/Hetu/tree/120b776d653708adfccbadc8e1b35d633eaf1161) with some extra optimization efforts:
* upgrading to CUDA 11
* adding RDMA support

## How to run Herald
### Prerequsites
See CMakeLists.txt for the required packages and libraries. In general, Herald requires
* cmake 3.18.0
* pybind11 2.6.1
* ZMQ 4.3.2
* OpenMPI 4.0.3
* protobuf 3.15.8
* NCCL 2.8

We also provide a [Dockerfile](docker/Dockerfile) to build the environment for Herald.

### Build HET and Herald
```bash
mkdir build
cd build
cmake ..
make -j
```

### Simple example
See the simple distributed configuration file [here](examples/config/dist.yml).

Before running the experiment, your need to correctly configure the `DMLC` and `NCCL` environment variables and other `mpi` related parameters in the `runner.py` and `run_hetu.py` files. For example, you need to set the `DMLC_INTERFACE` to the network interface that the workers and the parameter server can communicate with each other. You also need to set the `NCCL_SOCKET_IFNAME` to allow NCCL find the correct network interface.

```bash
# run hetu
python3 ./python/runner.py -c examples/config/dist.yml python3 ./examples/ctr/run_hetu.py --model wdl_criteo --comm Hybrid --cache lru --bound 0 --bsp 0 --nepoch 1 --all --batch-size 256 --embedding-size 512 --cache-limit-ratio 0.1

# run laia
python3 ./python/runner.py -c examples/config/dist.yml python3 ./examples/ctr/run_laia.py --model wdl_criteo --comm Hybrid --cache lru --bound 0 --bsp 0 --nepoch 1 --all --batch-size 256 --embedding-size 512 --cache-limit-ratio 0.1
```

## Reference

The design and implementation of Herald, as well as the experience results, has been documented in a paper accepted for publication at [NSDI'24](https://www.usenix.org/conference/nsdi24/presentation/zeng).
```
 @inproceedings{herald,
   title = {Accelerating Neural Recommendation Training with Embedding Scheduling},
   author = {Zeng, Chaoliang and
            Liao, Xudong and
            Cheng, Xiaodian and
            Tian, Han and
            Wan, Xinchen and
            Wang, Hao and
            Chen, Kai},
   booktitle = {NSDI},
   year = {2024}
 }
```