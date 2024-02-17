#include <algorithm>
#include <cassert>
#include <ios>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <chrono>
#include <unordered_set>

#include "array.h"
#include "laia_scheduler.h"
#include "utils.h"
#include "thread_pool.h"

using namespace std::chrono_literals;
namespace laia_cache {

LaiaScheduler::LaiaScheduler() : mut_(), pop_mut_(), data_cond_() {
}

LaiaScheduler::~LaiaScheduler() {
    if (dist_thread_) {
        this->close_ = true;
        dist_thread_->join();
    }
}

void LaiaScheduler::start(py::array_t<emb_key_t> sample_embs_py,
                          size_t num_sample, size_t num_table, size_t epoch_num,
                          size_t mini_batch_size, size_t batch_num,
                          size_t nrank, size_t rank, size_t cache_size,
                          size_t num_threads, size_t top_k_table) {
    if (sample_embs_py.ndim() != 2)
        throw runtime_error("Input should be 2D numpy array");
    assert(size_t(sample_embs_py.shape()[0]) == num_sample);
    assert(size_t(sample_embs_py.shape()[1]) == num_table);

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
    for (auto &cache : this->cache_snaps_)
        cache.set_cap(cache_size);
    // init unique keys
    this->unique_keys_.resize(nrank);

    this->epoch_id_ = 0;
    this->sample_embs_ = Array2D(2, {num_sample, num_table}, 0);
    cout << "inited all states" << endl;
    // copy sample_embs_py to 2Darray
    for (size_t i = 0; i < num_sample_; i++) {
        for (size_t j = 0; j < num_table_; j++) {
            sample_embs_[i][j] = sample_embs_py.at(i, j);
        }
    }
    // record table frequecies
    this->top_k_table_ = top_k_table;
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

    // start get_dist thread
    this->dist_thread_ = make_shared<std::thread>(
        std::thread(std::bind(&LaiaScheduler::launch, this)));
}

bool LaiaScheduler::queue_empty() {
    std::lock_guard<std::mutex> lg(this->mut_);
    return this->info_queue_.empty();
}

size_t LaiaScheduler::queue_length() {
    std::lock_guard<std::mutex> lg(this->mut_);
    return this->info_queue_.size();
}

void LaiaScheduler::push(const element_t& element) {
    std::lock_guard<std::mutex> lg(this->mut_);
    this->info_queue_.push(std::move(element));
    this->data_cond_.notify_one();
}

element_t LaiaScheduler::pop() {
    std::unique_lock<std::mutex> lg(this->pop_mut_);
    this->data_cond_.wait(lg, [this] { return !this->queue_empty(); });
    // push-pop mutex
    std::unique_lock<std::mutex> lock(this->mut_);
    auto res = std::move(this->info_queue_.front());
    this->info_queue_.pop();
    return res;
}

void LaiaScheduler::launch() {
    // partition data and get comm plan and distribution
    comm_plan_t cplan;
    dist_t dist(2, {num_worker_, mini_batch_size_}, 0);

    vector<set<emb_key_t>> unique_keys;
    cplan.resize(num_worker_);
    unique_keys.resize(num_worker_);
    auto thread_pool = ThreadPool::Get();
    cout << "start partition thread" << endl;
    cout << "epoch num: " << epoch_num_ << endl;

    while (epoch_id_ < epoch_num_ && !this->close_.load()) {
        batch_id_ = 0;
        epoch_id_++;

        // for the last epoch, we need one more allocation for cache prefetch
        if (epoch_id_ == epoch_num_) {
            batch_num_ += 1;
        }
        while (batch_id_ < batch_num_ && !this->close_.load()) {
            TIMING(get_dist(cplan, dist));

            push({cplan[rank_].begin(), cplan[rank_].end()});
            push({dist[rank_], dist[rank_] + mini_batch_size_});

            if (not thread_pool->empty()) {
                std::this_thread::sleep_for(1ms);
                thread_pool->clear();
            }

            for (size_t i = 0; i < num_worker_; i++) {
                thread_pool->Enqueue([&, i]() -> void {
                    for (auto key : cplan[i]) {
                        cache_snaps_[i].outdate(key);
                    }
                    unique_keys[i].clear();
                    for (size_t j = 0; j < mini_batch_size_; j++) {
                        for (size_t s = 0; s < num_table_; s++) {
                            unique_keys[i].emplace(sample_embs_[dist[i][j]][s]);
                        }
                    }
                    for (auto key : unique_keys[i]) {
                        cache_snaps_[i].get(key);
                    }
                });
            }
            thread_pool->Wait(num_worker_);

            batch_id_++;
        }
    }
    // notify python to end
    push({0});
}

void LaiaScheduler::get_dist(comm_plan_t &cplan, dist_t &dist) {
    auto thread_pool = ThreadPool::Get();

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

    // sort table as max frequecies
    // std::nth_element(table_frequency_.begin(),
    //                  table_frequency_.begin() + top_k_table_,
    //                  table_frequency_.end());

    // score
    for (size_t t = 0; t < num_thread_; t++) {
        thread_pool->Enqueue([&, t]() -> void {
            int x = batch_size_ / num_thread_;
            int y = batch_size_ % num_thread_;
            int start = (t == 0) ? 0 : (y + t * x);
            int end = (t == 0) ? (x + y) : (start + x);
            for (int i = start; i < end; i++) {
                for (size_t j = 0; j < num_table_; j++) {
                // for (int k = 0; k < top_k_table_; k++) {
                    // int j = table_frequency_[k].index;
                    auto emb = batch_sample[i][j];
                    // table_frequency_[k].record(emb);
                    bool find_flag = false;
                    for (size_t z = 0; z < num_worker_; z++) {
                        if (cache_snaps_[z].check(emb)) {
                            scores_[i][z] += 1;
                            sample_emb_dep_(i, z, count_[i][z]) = emb;
                            count_[i][z] += 1;
                            find_flag = true;
                        }
                    }
                    if (!find_flag) {
                        sample_emb_dep_(i, num_worker_,
                                        count_[i][num_worker_]) = emb;
                        count_[i][num_worker_] += 1;
                    }
                }
            }
        });
    }
    thread_pool->Wait(num_thread_);

    // distribution
    vector<unordered_set<emb_key_t>> dist_keys;
    dist_keys.resize(num_worker_);
    int max_workload = mini_batch_size_;
    dist.reset(0);
    for (size_t i = 0; i < batch_size_; i++) {
        int max_score = -1;
        int max_score_worker = -1;
        for (size_t j = 0; j < num_worker_; j++) {
            int worker = (j + batch_id_) % num_worker_;
            int score = scores_[i][worker];
            if (workers_workload_[worker] < max_workload) {
                if (max_score < score) {
                    max_score = score;
                    max_score_worker = worker;
                }
            }
        }
        dist[max_score_worker][workers_workload_[max_score_worker]] =
            (i + batch_start_idx) % num_sample_;
        dist_keys[max_score_worker].emplace((i + batch_start_idx)
                                            % num_sample_);
        workers_workload_[max_score_worker] += 1;
    }

    // communication plan
    for (size_t i = 0; i < num_worker_; i++) {
        boost::container::flat_set<emb_key_t> *plan_i;
        plan_i = &cplan[i];
        thread_pool->Enqueue([&, i, plan_i]() -> void {
            plan_i->clear();
            for (size_t s = 0; s < batch_size_; s++) {
                bool in_flag = false;
                auto position = (s + batch_start_idx) % num_sample_;
                in_flag = (dist_keys[i].find(position) != dist_keys[i].end());
                if (!in_flag) {
                    for (uint64_t j = 0; j < count_[s][i]; j++) {
                        auto emb = sample_emb_dep_(s, i, j);
                        plan_i->emplace(emb);
                    }
                }
            }
        });
    }
    thread_pool->Wait(num_worker_);
}

} // namespace laia_cache