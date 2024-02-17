#ifndef LAIA_SCHEDULER_H
#define LAIA_SCHEDULER_H

#include <algorithm>
#include <asm-generic/errno.h>
#include <atomic>
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

#include "mini_lru_cache.h"
#include "binding.h"
#include "utils.h"
#include "array.h"

using namespace std;

typedef uint64_t emb_key_t;

// using dist_t = Array2D;
// using comm_plan_t = vector<set<emb_key_t>>;
// using element_t = std::variant<comm_plan_t, dist_t>;
// using element_t = vector<emb_key_t>;

namespace laia_cache {
class LaiaScheduler {
public:
    static LaiaScheduler *Get() {
        static LaiaScheduler e;
        return &e;
    }

    LaiaScheduler();
    ~LaiaScheduler();

    void start(py::array_t<emb_key_t> samples_embs_np, size_t num_table,
               size_t num_sample, size_t epoch_num, size_t mini_batch_size,
               size_t batch_num, size_t nrank, size_t rank, size_t cache_size,
               size_t num_threads, size_t top_k_table);

    void get_dist(comm_plan_t &cplan, dist_t &dist);

    void launch();

    void push(const element_t& element);

    bool queue_empty();

    size_t queue_length();

    element_t pop();

private:
    class Table {
    public:
        struct TableElement {
            emb_key_t key {0};
            uint64_t occur_time {0};
            double frequency {0.0};
            TableElement(): key(0), occur_time(0), frequency(0.0){}
            TableElement(emb_key_t key, uint64_t occur_time) :
                key(key), occur_time(occur_time) {
            }

            TableElement(const TableElement &element) {
                key = element.key;
                occur_time = element.occur_time;
                frequency = element.frequency;
            }

            friend inline bool operator<(const TableElement &node1,
                                         const TableElement &node2) {
                return node1.frequency < node2.frequency;
            }
        };
        map<emb_key_t, TableElement> elements_;
        uint64_t total_occurance_;
        double max_frequency_;
        using pair_type = decltype(elements_)::value_type;
        int index;
        // TODO: deal with move constructor of mutex
        // mutable std::mutex mu_;

    public:
        Table() : total_occurance_(0), max_frequency_(0.0){}
        
        friend inline bool operator<(const Table &node1,
                                     const Table &node2) {
            return node1.max_frequency_ < node2.max_frequency_;
        }

        void record(emb_key_t key) {
            // std::lock_guard<std::mutex> gd(mu_);
            if (elements_.find(key) == elements_.end())
                elements_.insert({key, TableElement(key, 0)});
            auto &element = elements_[key];
            total_occurance_ += 1;
            element.occur_time += 1;
            element.frequency = double(element.occur_time) / total_occurance_;
            // update max frequency
            max_frequency_ = std::max(max_frequency_, element.frequency);
        }

        TableElement get_most_frequent_element() const {
             auto max = std::max_element(
                std::begin(elements_), std::end(elements_),
                [](const pair_type &p1, const pair_type &p2) {
                    return p1.second.frequency < p2.second.frequency;
                });
             return max->second;
        }
    };

private:
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
    vector<MiniLRUCache> cache_snaps_;
    vector<set<emb_key_t>> unique_keys_;

    size_t epoch_id_;
    size_t batch_id_;

    atomic<bool> close_{false};
    shared_ptr<thread> dist_thread_;
    mutable std::mutex mut_;
    mutable std::mutex pop_mut_;
    mutable std::condition_variable data_cond_;

    std::queue<element_t> info_queue_;

    int top_k_table_;
    vector<Table> table_frequency_;

    // varables for get_dist
    Array2D count_;
    Array3D sample_emb_dep_;
    Array2D scores_;
    vector<int> workers_workload_;
};

} // namespace laia_cache

#endif /* LAIA_SCHEDULER_H */