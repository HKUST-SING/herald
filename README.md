# Herald

This branch contains the RDMA support in Herald. We implemented the verbs-based RDMA transmission mechanism for embedding push/pull, which is able to achieve high link bandwidth utilization. The code is based on the RDMA library in [BytePS](https://github.com/bytedance/byteps).

Before running the code, you need to install the NIC driver and required RDMA libraries. Our code has been tested on NVIDIA Mellanox ConnectX-5 NICs with the MLNX_OFED driver on Ubuntu 18.04.

Here is an example of how to use Herald with RDMA support (add the `--rdma` flag):

```bash
python3 ./python/runner.py --rdma -c examples/config/dist.yml python3 ./examples/ctr/run_laia.py --model wdl_criteo --comm Hybrid --cache lru --bound 0 --bsp 0 --nepoch 1 --all --batch-size 256 --embedding-size 512 --cache-limit-ratio 0.1 --rdma
```
