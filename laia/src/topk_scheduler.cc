#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstddef>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_set>

#include <boost/container/flat_set.hpp>

#include "array.h"
#include "share_mem.h"
#include "thread_pool.h"
#include "topk_scheduler.h"
#include "utils.h"

using namespace std;
using namespace std::chrono_literals;
using namespace boost;

namespace laia_cache {

TopkScheduler::TopkScheduler() : mut_(), pop_mut_(), data_cond_() {
}

TopkScheduler::~TopkScheduler() {
    if (dist_thread_) {
        this->close_ = true;
        dist_thread_->join();
    }
}

void TopkScheduler::start(py::array_t<emb_key_t> sample_embs_py,
                          size_t num_sample, size_t num_table, size_t epoch_num,
                          size_t mini_batch_size, size_t batch_num,
                          size_t nrank, size_t rank, size_t cache_size,
                          size_t num_threads, const string &dataset,
                          size_t top_k_table, bool local_shared,
                          size_t local_rank, size_t local_size) {
    if (sample_embs_py.ndim() != 2)
        throw runtime_error("Input should be 2D numpy array");
    assert(size_t(sample_embs_py.shape()[0]) == num_sample);
    assert(size_t(sample_embs_py.shape()[1]) == num_table);

    // find max key in sample_embs_py
    emb_key_t max_key = 0;
    for (size_t i = 0; i < num_sample; i++) {
        for (size_t j = 0; j < num_table; j++) {
            max_key = max(max_key, sample_embs_py.at(i, j));
        }
    }

    // scheduler launched in standalone mode or local shared mode
    this->local_shared_ = local_shared;
    this->local_rank_ = local_rank;
    this->local_size_ = local_size;
    if (this->local_shared_) {
        // setup shared memory for local workers
        this->shared_mem_bufs_.resize(this->local_size_);
        if (local_rank == 0) {
            // create shared memory
            for (size_t i = 0; i < this->local_size_; i++) {
                this->shared_mem_bufs_[i] = std::make_shared<SharedMemBuf>(
                    "laia_cache_" + std::to_string(i), true, 1 << 30);
            }
            // open local shared memory
            this->local_shared_mem_buf_ = std::make_shared<SharedMemBuf>(
                "laia_cache_" + std::to_string(this->local_rank_), false);
        } else {
            // open shared memory
            this->local_shared_mem_buf_ = std::make_shared<SharedMemBuf>(
                "laia_cache_" + std::to_string(this->local_rank_), false);
        }
    }
    this->num_worker_ = nrank;
    this->rank_ = rank;
    this->batch_num_ = batch_num;
    this->epoch_num_ = epoch_num;
    this->cache_size_ = cache_size;
    this->mini_batch_size_ = mini_batch_size;
    this->batch_size_ = mini_batch_size * nrank;
    this->num_thread_ = num_threads;

    // num table & num samples
    this->num_table_ = num_table;
    this->num_sample_ = num_sample;

    this->cache_snaps_.resize(nrank);
    // for (size_t i = 0; i < nrank; i++) {
    //     this->cache_snaps_.emplace_back(
    //         MiniLRUCache(max_key+1));
    // }
    for (auto &cache : this->cache_snaps_)
        cache.set_cap(cache_size);
    // init unique keys
    this->unique_keys_.resize(nrank);

    this->epoch_id_ = 0;
    this->sample_embs_ = Array2D(2, {num_sample, num_table}, 0);
    // copy sample_embs_py to 2Darray
    for (size_t i = 0; i < num_sample_; i++) {
        for (size_t j = 0; j < num_table_; j++) {
            sample_embs_[i][j] = sample_embs_py.at(i, j);
        }
    }
    // record table frequecies
    this->top_k_table_ = top_k_table;
    if (!this->top_k_table_)
        this->top_k_table_ = num_table_;
    this->table_frequency_.resize(num_table);
    int index = 0;
    for (auto &table : table_frequency_) {
        table.index = index;
        index += 1;
    }
    // init
    count_ = Array2D(2, {batch_size_, num_worker_ + 1}, 0);
    sample_emb_dep_ = Array3D(3, {batch_size_, num_worker_ + 1, num_table_}, 0);
    scores_ = Array2D(2, {batch_size_, num_worker_}, 0);
    workers_workload_ = {0};
    workers_workload_.resize(num_worker_);

    // calculate cache push/pull
    miss_pull_.resize(num_worker_);
    miss_push_.resize(num_worker_);
    update_pull_.resize(num_worker_);
    update_push_.resize(num_worker_);

    write_locks.reserve(num_worker_);
    while (write_locks.size() < write_locks.capacity())
        write_locks.emplace_back(make_unique<std::mutex>());

    // init thread pool
    thread_pool_ = std::move(std::make_unique<ThreadPool>(num_threads));

    // criteo
    if (dataset == "criteo") {
        top_k_table_index_ = {9,  13, 22, 20, 12, 21, 17, 14, 24, 3, 5, 10, 16,
                              15, 19, 2,  4,  11, 7,  25, 23, 18, 8, 1, 0,  6};
    } else if (dataset == "avazu") {
        top_k_table_index_ = {1, 2,  4, 5,  15, 7, 6,  16, 12,
                              0, 17, 8, 14, 10, 9, 11, 13, 3};
    } else if (dataset == "movie") {
        top_k_table_index_ = {0, 1};
    } else if (dataset == "criteosearch") {
        top_k_table_index_ = {0,  11, 3, 4, 5,  14, 1, 6, 2,
                              13, 16, 9, 8, 10, 12, 7, 15};
    } else {
        std::cerr << "dataset not supported" << std::endl;
        exit(1);
    }

    this->top_k_table_ =
        std::min(this->top_k_table_, top_k_table_index_.size());

    if (!this->local_shared_ or this->local_rank_ == 0) {
        std::cout << "TopkScheduler: use top_k table: " << top_k_table_
                  << " for dataset: " << dataset
                  << ", worker num: " << num_worker_
                  << ", thread num: " << num_thread_ << endl;
    }

    // start get_dist thread
    if (!this->local_shared_ or this->local_rank_ == 0) {
        this->dist_thread_ = make_shared<std::thread>(
            std::thread(std::bind(&TopkScheduler::launch, this)));
    }
    // launch();
}

bool TopkScheduler::queue_empty() {
    std::lock_guard<std::mutex> lg(this->mut_);
    return this->info_queue_.empty();
}

size_t TopkScheduler::queue_length() {
    if (!this->local_shared_) {
        std::lock_guard<std::mutex> lg(this->mut_);
        return this->info_queue_.size();
    }
    return this->local_shared_mem_buf_->queue_length();
}

void TopkScheduler::push(const element_t &element) {
    std::lock_guard<std::mutex> lg(this->mut_);
    if (!this->local_shared_) {
        this->info_queue_.push(std::move(element));
        this->data_cond_.notify_one();
    }
}

void TopkScheduler::push_to_local_worker(const element_t &element,
                                         size_t local_rank) {
    assert(this->local_shared_);
    // only local rank 0 push to queue
    assert(this->local_rank_ == 0);
    while (1) {
        int ret = this->shared_mem_bufs_[local_rank]->send_data(element);
        if (ret > 0) {
            // write_count += ret;
            // std::cerr << "Rank 0 write to local worker " << local_rank
            //           << ", array len " << element.size() << ", ret " << ret
            //           << ", write count " << write_count << std::endl;
            break;
        } else
            std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

element_t TopkScheduler::pop() {
    std::unique_lock<std::mutex> lg(this->pop_mut_);
    this->data_cond_.wait(lg, [this] { return !this->queue_empty(); });
    // push-pop mutex
    std::unique_lock<std::mutex> lock(this->mut_);
    auto res = std::move(this->info_queue_.front());
    this->info_queue_.pop();
    return res;
}

element_t TopkScheduler::pop_from_local_worker() {
    assert(this->local_shared_);
    vector<uint64_t> res;
    // std::cerr << "read from local worker " << this->local_rank_ << std::endl;
    while (1) {
        res = this->local_shared_mem_buf_->recv_data();
        if (res.size() > 0) {
            // read_count += res.size();
            // if (this->local_rank_ != 0) {
            //     std::cerr << "Rank " << this->local_rank_
            //               << " read from local worker " << local_rank
            //               << ", array len " << res.size() << ", read count "
            //               << read_count << ". Read: [";
            // output first 5 elements
            //     for (int i = 0; i < 5; i++)
            //         std::cerr << res[i] << ", ";
            //     std::cerr << "]" << std::endl;
            // }
            break;
        } else
            std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    return res;
}

void TopkScheduler::launch() {
    // partition data and get comm plan and distribution
    comm_plan_t cplan;
    dist_t dist(2, {num_worker_, mini_batch_size_}, 0);

    // vector<set<emb_key_t>> unique_keys;
    vector<container::flat_set<emb_key_t>> unique_keys;
    cplan.resize(num_worker_);
    unique_keys.resize(num_worker_);
    cout << "start partition thread, epoch num: " << epoch_num_ << endl;

    while (epoch_id_ < epoch_num_ && !this->close_.load()) {
        batch_id_ = 0;
        epoch_id_++;

        // for the last epoch, we need one more allocation for cache prefetch
        if (epoch_id_ == epoch_num_) {
            batch_num_ += 1;
        }
        std::chrono::system_clock::time_point iter_start, iter_end;
        while (batch_id_ < batch_num_ && !this->close_.load()) {
            if (batch_id_ >= 12) {
                iter_start = std::chrono::high_resolution_clock::now();
            }
            get_dist(cplan, dist);

            if (not this->local_shared_) {
                push({cplan[rank_].begin(), cplan[rank_].end()});
                push({dist[rank_], dist[rank_] + mini_batch_size_});
            } else {
                assert(this->local_rank_ == 0);
                for (size_t i = 0; i < local_size_; i++) {
                    // get the information according to global rank
                    auto worker_id = this->rank_ + i;
                    push_to_local_worker(
                        {cplan[worker_id].begin(), cplan[worker_id].end()}, i);
                    push_to_local_worker(
                        {dist[worker_id], dist[worker_id] + mini_batch_size_},
                        i);
                }
            }

            // if (not thread_pool_->empty()) {
            //     std::this_thread::sleep_for(1ms);
            //     thread_pool_->clear();
            // }

            for (size_t i = 0; i < num_worker_; i++) {
                thread_pool_->Enqueue([&, i]() -> void {
                    try {
                        for (auto key : cplan[i]) {
                            cache_snaps_[i].outdate(key);
                        }
                        unique_keys[i].clear();
                        for (size_t j = 0; j < mini_batch_size_; j++) {
                            for (size_t s = 0; s < num_table_; s++) {
                                unique_keys[i].emplace(
                                    sample_embs_[dist[i][j]][s]);
                            }
                        }
                        for (auto key : unique_keys[i]) {
                            int res = cache_snaps_[i].get(key);
                            if (res < 0) {
                                if (res == -2) {
                                    update_pull_[i] += 1;
                                }
                            } else {
                                miss_pull_[i] += 1;
                                if (res > 0)
                                    miss_push_[i] += 1;
                            }
                        }
                        update_push_[i] += cplan[i].size();
                    } catch (std::exception &e) {
                        cerr << "Exception: " << e.what() << endl;
                    }
                });
            }
            thread_pool_->Wait(num_worker_);
            if (batch_id_ >= 12) {
                iter_end = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        iter_end - iter_start);
                iter_time_.push_back(duration.count());
            }
            batch_id_++;
        }
    }
    // notify python to end
    if (not this->local_shared_) {
        push({0});
    } else {
        assert(this->local_rank_ == 0);
        for (size_t i = 0; i < local_size_; i++) {
            push_to_local_worker({0}, i);
        }
    }
    complete_.store(true);
    complete_cond_.notify_one();
    cout << "get dist thread done" << endl;
}

void TopkScheduler::get_dist(comm_plan_t &cplan, dist_t &dist) {
    // reset variables
    count_.reset(0);
    scores_.reset(0);
    sample_emb_dep_.reset(0);
    workers_workload_.clear();
    workers_workload_.resize(num_worker_);

    // current batch
    int batch_start_idx = (batch_id_ * batch_size_) % num_sample_;
    vector<emb_key_t *> batch_sample;
    for (size_t i = batch_start_idx; i < batch_start_idx + batch_size_; i++) {
        batch_sample.emplace_back(sample_embs_[i % num_sample_]);
    }

    // score
    vector<int> mscore_worker(batch_size_, 0);
    vector<uint64_t> max_score(batch_size_, 0);

    // distribution
    int max_workload = mini_batch_size_;
    dist.reset(0);

    // vector<vector<int>> all_worker_workload;
    // all_worker_workload.resize(num_thread_);

    vector<vector<size_t>> all_workers_workload(num_thread_);
    for (auto &w : all_workers_workload) {
        w.resize(num_worker_, 0);
    }

    for (size_t t = 0; t < num_thread_; t++) {
        // auto &workers_workload = all_worker_workload[t];
        // workers_workload.resize(num_worker_);
        thread_pool_->Enqueue([&, t]() -> void {
            auto &workers_workload = all_workers_workload[t];
            int x = batch_size_ / num_thread_;
            int y = batch_size_ % num_thread_;
            int start = (t == 0) ? 0 : (y + t * x);
            int end = (t == 0) ? (x + y) : (start + x);

            int wx = max_workload / num_thread_;
            int wy = max_workload % num_thread_;
            int wstart = (t == 0) ? 0 : (wy + t * wx);
            int wend = (t == 0) ? (wx + wy) : (wstart + wx);
            size_t local_workload = wend - wstart;

            for (int i = start; i < end; i++) {
                for (size_t k = 0; k < top_k_table_; k++) {
                    // int j = table_frequency_[k].index;
                    int j = top_k_table_index_[k];
                    // int j = 0;
                    auto emb = batch_sample[i][j];
                    // table_frequency_[k].record(emb);
                    // bool find_flag = false;
                    for (size_t z = 0; z < num_worker_; z++) {
                        if (cache_snaps_[z].check(emb)) {
                            scores_[i][z] += 1;
                            // record max_score and max_score_worker
                            if (scores_[i][z] > max_score[i]) {
                                max_score[i] = scores_[i][z];
                                mscore_worker[i] = z;
                            }
                        }
                    }
                }
                int max_score = -1;
                int max_score_worker = -1;
                int candidate = mscore_worker[i];
                bool find_flag = false;
                for (size_t j = 0; j < num_worker_ && !find_flag; j++) {
                    // int worker = (j + t) % num_worker_;
                    int worker = (j + candidate) % num_worker_;
                    int score = scores_[i][worker];
                    if (max_score < score
                        && workers_workload[worker] < local_workload) {
                        max_score = score;
                        max_score_worker = worker;
                        if (max_score_worker == candidate)
                            find_flag = true;
                    }
                }
                {
                    // write lock
                    // std::unique_lock<std::mutex> lg(
                    //     *write_locks[max_score_worker]);
                    dist[max_score_worker]
                        [wstart + workers_workload[max_score_worker]] =
                            (i + batch_start_idx) % num_sample_;
                    workers_workload[max_score_worker] += 1u;
                    // write lock
                }
            }
        });
    }
    thread_pool_->Wait(num_thread_);

    // merge
    // for (size_t t = 0; t < num_thread_; t++) {
    //     auto &dist_keys = all_dist_keys[t];
    //     for (size_t i = 0; i < num_worker_; i++) {
    //         dist_keys_[i].emplace(dist_keys[i].begin(), dist_keys[i].end())
    //     }
    // }

    // communication plan
    for (size_t i = 0; i < num_worker_; i++) {
        auto samples = make_shared<vector<emb_key_t>>();
        samples->reserve(batch_size_ * num_table_);
        auto my_dist =
            make_shared<vector<emb_key_t>>(dist[i], dist[i] + mini_batch_size_);
        thread_pool_->Enqueue([&, i, samples, my_dist]() -> void {
            if (enable_comm_plan_) {
                for (size_t s = 0; s < batch_size_; s++) {
                    auto position = (s + batch_start_idx) % num_sample_;
                    if (std::find(my_dist->begin(), my_dist->end(), position)
                        != my_dist->end()) {
                        samples->insert(samples->end(), batch_sample[s],
                                        batch_sample[s] + num_table_);
                    }
                }
                auto tmp_set = make_unique<container::flat_set<emb_key_t>>(
                    samples->begin(), samples->end());
                for (auto &key : *tmp_set) {
                    if (!cache_snaps_[i].check(key)) {
                        tmp_set->erase(key);
                    }
                }
                cplan[i] = std::move(*tmp_set);
            } else {
                // for (auto sample_index : dist_keys[i]) {
                //     for (size_t j = 0; j < num_table_; j++) {
                //         auto emb = sample_embs_[sample_index][j];
                //         plan_i->emplace(emb);
                //     }
                // }
            }
        });
    }
    thread_pool_->Wait(num_worker_);
}

map<string, long> TopkScheduler::report_cache_perf() {
    // wait until complete
    std::unique_lock<std::mutex> lg(complete_mut_);
    complete_cond_.wait(lg, [this]() -> bool { return complete_; });

    map<string, long> results;
    int avg_miss_pull =
        std::accumulate(miss_pull_.begin(), miss_pull_.end(), 0) / num_worker_;
    int avg_miss_push =
        std::accumulate(miss_push_.begin(), miss_push_.end(), 0) / num_worker_;
    int avg_update_pull =
        std::accumulate(update_pull_.begin(), update_pull_.end(), 0)
        / num_worker_;
    int avg_update_push =
        std::accumulate(update_push_.begin(), update_push_.end(), 0)
        / num_worker_;

    results.insert({"miss_pull", avg_miss_pull});
    results.insert({"miss_push", avg_miss_push});
    results.insert({"update_pull", avg_update_pull});
    results.insert({"update_push", avg_update_push});

    return results;
}

void TopkScheduler::reset_scheduler() {
    // first wait dist thread to complete
    if (dist_thread_) {
        dist_thread_->join();
        dist_thread_ = nullptr;
    }
    for (auto &cache : this->cache_snaps_)
        cache.clear();
    workers_workload_ = {0};
    epoch_id_ = 0;
    while (info_queue_.size() > 0) {
        info_queue_.pop();
    }

    complete_ = false;
    miss_pull_.clear();
    miss_push_.clear();
    update_pull_.clear();
    update_push_.clear();

    close_.store(false);
}

map<string, long> TopkScheduler::report_overhead() {
    map<string, long> results;
    return results;
}

long TopkScheduler::report_iter_time() {
    // wait until complete
    std::unique_lock<std::mutex> lg(complete_mut_);
    complete_cond_.wait(lg, [this]() -> bool { return complete_; });
    if (iter_time_.size() == 0)
        return 0;
    cout << "iter time size: " << iter_time_.size() << ", [";
    for (auto &t : iter_time_) {
        cout << t << ", ";
    }
    cout << "]" << endl;
    auto iter_time = std::accumulate(iter_time_.begin(), iter_time_.end(), 0)
                     / iter_time_.size();
    return iter_time;
}

} // namespace laia_cache