#ifndef TOPK_SCHEDULER_H
#define TOPK_SCHEDULER_H

#include <algorithm>
#include <asm-generic/errno.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>
#include <set>
#include <thread>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <utility>
#include <variant>
// #include <semaphore>
#include <boost/container/flat_set.hpp>

#include "mini_lru_cache.h"
#include "binding.h"
#include "thread_pool.h"
#include "utils.h"
#include "array.h"
#include "share_mem.h"

using namespace std;
using namespace boost;

namespace laia_cache {
class TopkScheduler {
public:
    static TopkScheduler *Get() {
        static TopkScheduler e;
        return &e;
    }

    TopkScheduler();
    ~TopkScheduler();

    void start(py::array_t<emb_key_t> samples_embs_np, size_t num_table,
               size_t num_sample, size_t epoch_num, size_t mini_batch_size,
               size_t batch_num, size_t nrank, size_t rank, size_t cache_size,
               size_t num_threads, const string &dataset, size_t top_k_table,
               bool local_shared, size_t local_rank, size_t local_size);

    void get_dist(comm_plan_t &cplan, dist_t &dist);

    void launch();

    void push(const element_t &element);

    void push_to_local_worker(const element_t &element, size_t local_rank);

    bool queue_empty();

    size_t queue_length();

    element_t pop();

    element_t pop_from_local_worker();

    void reset_scheduler();

    map<string, long> report_cache_perf();

    map<string, long> report_overhead();

    long report_iter_time();

private:
    class Table {
    public:
        struct TableElement {
            emb_key_t key{0};
            uint64_t occur_time{0};
            double frequency{0.0};
            TableElement() : key(0), occur_time(0), frequency(0.0) {
            }
            TableElement(emb_key_t key, uint64_t occur_time) :
                key(key), occur_time(occur_time) {
            }

            TableElement(const TableElement &element) {
                key = element.key;
                occur_time = element.occur_time;
                frequency = element.frequency;
            }

            TableElement(TableElement &&element) {
                key = element.key;
                occur_time = element.occur_time;
                frequency = element.frequency;
            }

            friend inline bool operator<(const TableElement &node1,
                                         const TableElement &node2) {
                return node1.frequency < node2.frequency;
            }
        };

        uint64_t total_occurance_;
        uint64_t max_frequency_;
        int index;
        mutable unique_ptr<std::mutex> mu_{nullptr};
        std::shared_ptr<map<emb_key_t, TableElement>> elements_;
        using pair_type = map<emb_key_t, TableElement>::value_type;

    public:
        Table() :
            total_occurance_(0), max_frequency_(0),
            mu_(std::make_unique<std::mutex>()),
            elements_(std::make_shared<map<emb_key_t, TableElement>>()) {
        }

        Table(const Table &table) {
            elements_ = std::move(table.elements_);
            total_occurance_ = table.total_occurance_;
            max_frequency_ = table.max_frequency_;
            if (!mu_)
                mu_ = std::make_unique<std::mutex>();
            // if (mu_)
            // mu_.reset();
        }

        Table &operator=(const Table &table) {
            elements_ = std::move(table.elements_);
            total_occurance_ = table.total_occurance_;
            max_frequency_ = table.max_frequency_;
            // if (mu_)
            // mu_.reset();
            // mu_ = std::make_unique<std::mutex>();
            return *this;
        }

        Table &operator=(Table &&table) {
            elements_ = std::move(table.elements_);
            total_occurance_ = table.total_occurance_;
            max_frequency_ = table.max_frequency_;
            mu_ = std::move(table.mu_);
            return *this;
        }

        // Table(Table&& table) {
        //     elements_ = std::move(table.elements_);
        //     total_occurance_ = table.total_occurance_;
        //     max_frequency_ = table.max_frequency_;
        //     if (mu_)
        //         mu_.reset();
        //     mu_ = std::move(table.mu_);
        // }

        friend inline bool operator<(const Table &node1, const Table &node2) {
            return node1.max_frequency_ < node2.max_frequency_;
        }

        void record(emb_key_t key) {
            // std::lock_guard<std::mutex> gd(*mu_);
            if (elements_->find(key) == elements_->end()) {
                std::lock_guard<std::mutex> gd(*mu_);
                if (elements_->find(key) == elements_->end())
                    elements_->insert({key, TableElement(key, 0)});
            }
            auto &element = (*elements_)[key];
            element.occur_time += 1;
            max_frequency_ = std::max(max_frequency_, element.occur_time);
            // total_occurance_ += 1;
            // element.occur_time += 1;
            // element.frequency = double(element.occur_time) /
            // total_occurance_;
            // // update max frequency
            // max_frequency_ = std::max(max_frequency_, element.frequency);
        }

        TableElement get_most_frequent_element() const {
            auto max = std::max_element(
                std::begin(*elements_), std::end(*elements_),
                [](const pair_type &p1, const pair_type &p2) {
                    return p1.second.frequency < p2.second.frequency;
                });
            return max->second;
        }
    };

    int partition(vector<Table> &tables, int start, int end);

    void find_mink_table(vector<Table> &tables, int k);

private:
    // standalone mode or local_shared mode
    bool local_shared_;
    size_t local_size_;
    size_t local_rank_;
    // used by local major worker (local_rank = 0)
    vector<std::shared_ptr<SharedMemBuf>> shared_mem_bufs_;
    // used by other local workers
    std::shared_ptr<SharedMemBuf> local_shared_mem_buf_;
    size_t epoch_num_;
    size_t num_sample_;
    size_t num_table_;
    size_t batch_num_;
    size_t cache_size_;
    // per worker
    size_t mini_batch_size_;
    // global
    size_t batch_size_;

    size_t num_thread_;
    size_t num_worker_;
    size_t rank_;
    Array2D sample_embs_;
    vector<MiniLRUCache> cache_snaps_{};
    vector<container::flat_set<emb_key_t>> unique_keys_;

    size_t epoch_id_;
    size_t batch_id_;

    atomic<bool> close_{false};
    std::shared_ptr<thread> dist_thread_;
    mutable std::mutex mut_;
    mutable std::mutex pop_mut_;
    mutable std::condition_variable data_cond_;

    std::queue<element_t> info_queue_;

    size_t top_k_table_;
    vector<Table> table_frequency_;

    // varables for get_dist
    Array2D count_;
    Array3D sample_emb_dep_;
    Array2D scores_;
    vector<int> workers_workload_{0};

    vector<int> miss_pull_{0};
    vector<int> miss_push_{0};
    vector<int> update_pull_{0};
    vector<int> update_push_{0};

    // factor analysis
    bool enable_score_{true};
    bool enable_comm_plan_{true};
    atomic<bool> complete_{false};
    mutable std::mutex complete_mut_;
    mutable std::condition_variable complete_cond_;

    unique_ptr<ThreadPool> thread_pool_{nullptr};

    vector<long> iter_time_;
    vector<unique_ptr<std::mutex>> write_locks;
    vector<int> top_k_table_index_;
};

} // namespace laia_cache

#endif /* TOPK_SCHEDULER_H */