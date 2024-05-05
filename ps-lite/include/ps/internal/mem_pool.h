#pragma once

#include "common/sarray.h"
#include "common/thread_safe_hash_map.h"
#include "ps/internal/postoffice.h"
#include "ps/psf/PSFunc.h"
#include "ps/psf/serializer.h"
#include "ps/kvapp.h"
#include "ps/partitioner.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <string>
#include <set>
#include <shared_mutex>

namespace ps {

#define MEM_BUF_SCALE 100

class SimpleMemPool {
public:
    SimpleMemPool() {
    }
    ~SimpleMemPool() {
        for (auto &iter : per_thread_mem_buf) {
            auto mem_buf = iter.second;
            for (auto &iter : mem_buf) {
                for (auto &mem : iter.second) {
                    free(mem.first);
                }
            }
        }
    }

public:
    void *getMemBuf(PsfType ftype, size_t size, int tid = 0) {
        // std::lock_guard<std::mutex> lock(mem_mtx);
        // mem buffer per thread
        // LOG(INFO) << "Get memory for " << ftype << " for thread " << tid;
        bool found = false;
        // ThreadMemPool::mapped_type *thread_mem_buf;
        {
            ReadLock lock(mem_mtx);
            found = per_thread_mem_buf.find(tid) != per_thread_mem_buf.end();
        }
        if (!found) {
            WriteLock lock(mem_mtx);
            per_thread_mem_buf[tid] = TypedMemPool();
            per_thread_used_mem[tid] = {};
        }
        // the following code should be thread safe
        auto &thread_mem_buf = per_thread_mem_buf[tid];
        auto &thread_used = per_thread_used_mem[tid];
        if (thread_mem_buf.find(ftype) == thread_mem_buf.end()) {
            thread_mem_buf[ftype] = OrderedMemList();
        }
        for (auto iter = thread_mem_buf[ftype].begin();
             iter != thread_mem_buf[ftype].end(); ++iter) {
            auto key = reinterpret_cast<uint64_t>(iter->first);
            auto len = iter->second;
            if (len >= size
                && (thread_used.find(key) == thread_used.end()
                    || !thread_used[key])) {
                // mark as used
                thread_used[key] = true;
                PS_VLOG(3) << "Reuse memory " << key;
                return iter->first;
            }
        }
        // if not found with adequate size, allocate a new one
        auto ptr = malloc(size);
        CHECK(ptr);
        thread_mem_buf[ftype].push_back(std::make_pair(ptr, size));
        auto ret = thread_mem_buf[ftype].back().first;
        // mark as used
        thread_used[reinterpret_cast<uint64_t>(ret)] = true;
        PS_VLOG(3) << "Allocate new memory for " << ftype << " at "
                   << reinterpret_cast<uint64_t>(ptr) << " size " << size;
        return ret;
    }

    bool releaseMem(uint64_t key, int tid = 0, bool is_worker = false) {
        if (!key)
            return true;
        // std::lock_guard<std::mutex> lock(mem_mtx);
        ThreadUsedMemList::iterator used_iter = per_thread_used_mem.end();
        {
            ReadLock lock(mem_mtx);
            if (per_thread_mem_buf.find(tid) == per_thread_mem_buf.end()) {
                LOG(FATAL) << "No Memory Pool for thread: " << tid;
                return false;
            }
            used_iter = per_thread_used_mem.find(tid);
            if (used_iter == per_thread_used_mem.end()) {
                LOG(FATAL) << "No Used Memory List for thread: " << tid;
                return false;
            }
        }
        if (is_worker) {
            std::lock_guard<std::mutex> lock(worker_mem_mtx);
            return releaseHelper(used_iter, key, tid);
        } else {
            return releaseHelper(used_iter, key, tid);
        }
    }

    std::string DebugString() {
        // for safety, we use a lock
        WriteLock lock(mem_mtx);
        std::stringstream ss;
        ss << "Memory Pool: ";
        for (auto &iter : per_thread_mem_buf) {
            ss << "Thread " + std::to_string(iter.first) << ": {";
            auto mem_buf = iter.second;
            for (auto &iter : mem_buf) {
                ss << "PsfType " + std::to_string(iter.first) << ": [";
                for (auto &mem : iter.second) {
                    ss << "(" << reinterpret_cast<uint64_t>(mem.first) << ", "
                       << mem.second << "), ";
                }
                ss << "], ";
            }
            ss << "}, ";
        }

        ss << ". Used Mem List: ";
        for (auto &iter : per_thread_used_mem) {
            ss << "Thread " + std::to_string(iter.first) << ": {";
            auto used_mem = iter.second;
            for (auto &iter : used_mem) {
                ss << "(" << iter.first << ", " << iter.second << "), ";
            }
            ss << "}, ";
        }
        return ss.str();
    }

private:
    typedef vector<std::pair<void *, size_t>> OrderedMemList;
    typedef std::unordered_map<PsfType, OrderedMemList> TypedMemPool;
    typedef std::unordered_map<int, TypedMemPool> ThreadMemPool;
    typedef std::unordered_map<int, std::unordered_map<uint64_t, bool>>
        ThreadUsedMemList;
    ThreadMemPool per_thread_mem_buf;
    // thread id -> used_mem_list
    ThreadUsedMemList per_thread_used_mem;
    std::shared_mutex mem_mtx;
    std::mutex worker_mem_mtx;

    typedef std::shared_lock<std::shared_mutex> ReadLock;
    typedef std::unique_lock<std::shared_mutex> WriteLock;

    bool releaseHelper(ThreadUsedMemList::iterator &used_iter, uint64_t key,
                       int tid = 0) {
        auto used_mem = (used_iter->second).find(key);
        if (used_mem != used_iter->second.end()) {
            if (used_mem->second) {
                PS_VLOG(3) << "release memory " << key;
                used_mem->second = false;
                return true;
            }
            // TODO if the flag is false, should we just panic?
            return true;
        } else {
            LOG(WARNING) << "Memory not found in used list: " << key;
            for (auto &iter : per_thread_mem_buf[tid]) {
                for (auto &mem : iter.second) {
                    if (reinterpret_cast<uint64_t>(mem.first) == key) {
                        LOG(WARNING)
                            << "Memory actually found in mem buf: " << key;
                        return false;
                    }
                }
            }
            return false;
        }
    }
};

} // namespace ps
