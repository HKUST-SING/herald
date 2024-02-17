from __future__ import absolute_import
import numpy as np
import sys
import os
import time

from .. import ndarray
from ..dataloader import Dataloader, DataloaderOp
from ..gpu_ops.Node import Op

current_dir = os.path.realpath(os.path.dirname(__file__))
build_dir = os.path.realpath(
    os.path.join(current_dir, os.pardir, os.pardir, os.pardir, "build", "lib")
)
sys.path.append(build_dir)
from laia_cache import LaiaScheduler
from laia_cache import TopkScheduler

top_k_table = {
    "criteo": 20,
    "avazu": 17,
    "movie": 2,
    "criteosearch": 16,
}

local_worker_num = 8


class LAIAScheduler(object):
    def __init__(self, sparse_data, batch_size, drop_last=True, dataset="criteo", local_shared=False):
        self.sparse_data = np.array(sparse_data, np.float32).astype(np.intc)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.init = False
        self.dataset = dataset
        self.local_shared = local_shared

    def start(self, config, dataset_num=3, epoch_num=-1):
        # assert config.nrank > 1, "LAIA does not support nrank <= 1, current nrank = {}".format(config.nrank)
        assert not self.init, "LAIA scheduler can only be initialized once"

        self.local_rank = config.local_rank
        self.samples_num = len(self.sparse_data) // config.nrank
        self.queue_size = 5
        self.batch_size = min(int(self.batch_size), self.samples_num // self.queue_size)
        assert self.batch_size > 0, "Batch size %d invalid." % self.batch_size
        self.batch_num = (
            int(np.ceil(self.samples_num / self.batch_size))
            if not self.drop_last
            else self.samples_num // self.batch_size
        )

        # Start laia scheduler in backend
        self.sched = None
        if not self.local_shared:
            self.sched = LaiaScheduler()
            self.sched.start(
                self.sparse_data,
                self.sparse_data.shape[0],
                self.sparse_data.shape[1],
                epoch_num,
                self.batch_size,
                self.batch_num,
                int(config.nrank),
                int(config.rank),
                int(config.cache_limit),
                16,
                # top_K_table, not used yet
                24,
            )
        else:
            # only local rank 0 start laia scheduler
            self.sched = TopkScheduler()
            # allow local major worker to start first
            if self.local_rank != 0:
                time.sleep(3)
            self.sched.start(
                self.sparse_data,
                self.sparse_data.shape[0],
                self.sparse_data.shape[1],
                epoch_num,
                self.batch_size,
                self.batch_num,
                int(config.nrank),
                int(config.rank),
                int(config.cache_limit),
                int(80),
                # top_K_table, pre-profiled offline
                self.dataset,
                int(top_k_table[self.dataset]),
                True,
                int(config.local_rank),
                # local size
                int(local_worker_num),
            )

        self.channel_close = False

        # input_index[i] is an array of input index
        self.input_index = []
        # comm_plan[i] is the corresponding communication plan for input_index[i]
        self.comm_plan = []
        # key: batch_id
        # value: index of input_index & comm_plan
        self.arr_map = {}

        # prefetch to fill up the queue
        for i in range(self.queue_size):
            if i == 0:
                # discard the first comm_plan
                self._channel_get()
            self.input_index.append(self._channel_get())
            self.comm_plan.append(self._channel_get())
            self.arr_map[i] = i

        self.step = [0] * dataset_num
        self.cur_min_step = 0

        self.init = True
        print("LAIA scheduler started")

    def _channel_get(self):
        if self.channel_close:
            raise RuntimeError(
                "Channle have been closed, but still try to get value from it"
            )

        if not self.local_shared:
            res = self.sched.pop()
        else:
            res = self.sched.pop_from_local_worker()

        assert isinstance(
            res, list
        ), "Channel return value ({}) with type ({}) invalid".format(
            res, str(type(res))
        )

        if len(res) == 1 and res[0] == 0:
            self.channel_close = True
            return []
        else:
            return res

    def get_input_index(self, batch_id):
        return self.input_index[self.arr_map[batch_id]]

    def get_comm_plan(self, batch_id):
        return self.comm_plan[self.arr_map[batch_id]]

    # NOTE: step_forward() is required after calling get_input_index and get_comm_plan
    def step_forward(self, dataset_id):
        self.step[dataset_id] += 1
        new_min_step = min(self.step)
        # replace the oldest batch with a new one
        while self.cur_min_step < new_min_step:
            # aovid waiting
            if self.channel_close or (
                self.sched.length() < 2
                and new_min_step - self.cur_min_step < self.queue_size
            ):
                break
            min_batch_id = self.cur_min_step % self.batch_num
            arr_index = self.arr_map.pop(min_batch_id)
            self.input_index[arr_index] = self._channel_get()
            self.comm_plan[arr_index] = self._channel_get()
            new_batch_id = (min_batch_id + self.queue_size) % self.batch_num
            self.arr_map[new_batch_id] = arr_index
            self.cur_min_step += 1


class LAIADataloader(Dataloader):
    def __init__(
        self,
        sched,
        sched_id,
        is_sparse,
        raw_data,
        batch_size,
        name="default",
        func=None,
        drop_last=True,
    ):
        super().__init__(raw_data, batch_size, name, func, drop_last)
        self.sched = sched
        self.sched_id = sched_id
        self.is_sparse = is_sparse

    def init_states(self, rank=None, nrank=None):
        if nrank is None:
            nrank = 1
        # assert nrank > 1, "LAIA does not support nrank <= 1, current nrank = {}".format(nrank)
        # self.samples_num = self.raw_data.shape[0] // nrank
        self.samples_num = self.sched.samples_num
        # self.batch_num = int(np.ceil(self.samples_num / self.batch_size)) if not self.drop_last else \
        #     self.samples_num // self.batch_size
        self.batch_num = self.sched.batch_num
        self.batch_size = self.sched.batch_size
        self.batch_index = 0
        self.rank = rank

    def _get_arr(self, batchind):
        # get specific batch
        idx = self.sched.get_input_index(self.batch_index)
        if self.is_sparse:
            return (
                ndarray.array(self.raw_data[idx], ctx=ndarray.cpu(0)),
                ndarray.array(
                    self.sched.get_comm_plan(self.batch_index), ctx=ndarray.cpu(0)
                ),
            )
        else:
            return ndarray.array(self.raw_data[idx], ctx=ndarray.cpu(0))

    def get_arr(self):
        # step forward in this function
        res = self._get_arr(self.batch_index)
        self.batch_index = (self.batch_index + 1) % self.batch_num
        self.sched.step_forward(self.sched_id)
        return res

    def get_next_arr(self):
        res = self._get_arr(self.batch_index)
        return res

    def get_cur_shape(self):
        return tuple(
            [len(self.sched.get_input_index(self.batch_index))]
            + list(self.raw_data.shape[1:])
        )


def laia_dataloader_op(dataloaders, sched, sched_id, is_sparse=False):
    """
    dataloaders: list of dataloaders
    """
    temp_dataloaders = []
    for dl in dataloaders:
        if isinstance(dl, Dataloader):
            temp_dataloaders.append(dl)
        elif isinstance(dl, list):
            if len(dl) >= 3 and dl[2] == "train":
                dl.insert(0, is_sparse)
                dl.insert(0, sched_id)
                dl.insert(0, sched)
                temp_dataloaders.append(LAIADataloader(*dl))
            else:
                temp_dataloaders.append(Dataloader(*dl))
        elif isinstance(dl, dict):
            if "train" == dl["name"]:
                dl["sched"] = sched
                dl["sched_id"] = sched_id
                dl["is_sparse"] = is_sparse
                temp_dataloaders.append(LAIADataloader(**dl))
            else:
                temp_dataloaders.append(Dataloader(**dl))
        else:
            assert False, "Dataloader parameter invalid."
    return DataloaderOp(temp_dataloaders)
