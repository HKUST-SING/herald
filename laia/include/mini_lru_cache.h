#ifndef MINILRUCACHE_H
#define MINILRUCACHE_H

#include <cstddef>
#include <list>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>

using namespace std;

namespace laia_cache {
class MiniLRUCache {
private:
    list<int> list_;
    unordered_map<int, pair<list<int>::iterator, bool>> hash_;
    vector<bool> bitmap_;
    unsigned int capacity_;
    size_t emb_size_;
    bool used_bitmap_;

public:
    // MiniLRUCache(unsigned int capacity) : capacity_(capacity) {}
    // MiniLRUCache(int capacity) : capacity_(capacity) {}
    MiniLRUCache(size_t emb_size): emb_size_(emb_size), used_bitmap_(true) {
        bitmap_.resize(emb_size_, false);
    }

    MiniLRUCache():emb_size_(0), used_bitmap_(false) {}

    // move constructor
    MiniLRUCache(MiniLRUCache &&other) {
        list_ = move(other.list_);
        hash_ = move(other.hash_);
        bitmap_ = move(other.bitmap_);
        capacity_ = other.capacity_;
        emb_size_ = other.emb_size_;
        used_bitmap_ = other.used_bitmap_;
    }

    void clear() {
        list_.clear();
        hash_.clear();
        if (used_bitmap_)
            bitmap_.resize(emb_size_, false);
    }

    void set_cap(int capacity) {
        capacity_ = capacity;
        hash_.reserve(capacity*3);
    }

    bool check(const int key) const {
        if (used_bitmap_)
            return bitmap_[key];
        auto it = hash_.find(key);
        if (it == hash_.end() || it->second.second == false) {
            return false;
        } else {
            return true;
        }
    }

    bool is_full() const {
        return hash_.size() == capacity_;
    }

    int get(int key) {
        auto it = hash_.find(key);
        if (it == hash_.end()) {
            // key not in the cache
            return insert(key);
        } else {
            auto cit = it->second.first;
            // if key is outdated, need update_pull
            int res = it->second.second ? -1 : -2;
            list_.erase(cit);
            list_.push_front(key);
            it->second = make_pair(list_.begin(), true);
            if (used_bitmap_)
                bitmap_[key] = true;
            return res;
        }
    }

    // ensure that key does not exist in hash before insertion
    int insert(int key) {
        list_.push_front(key);
        hash_[key] = make_pair(list_.begin(), true);
        if (used_bitmap_)
            bitmap_[key] = true;
        if (hash_.size() > capacity_) {
            int ekey = list_.back();
            bool flag = hash_[ekey].second;
            hash_.erase(ekey);
            list_.pop_back();
            if (used_bitmap_)
                bitmap_[ekey] = false;
            // if poped key is new, need miss_push
            int res = flag ? 1 : 0;
            return res;
        }
        return 0;
    }

    void evict(int key) {
        auto it = hash_.find(key);
        if (it != hash_.end()) {
            auto cit = it->second.first;
            list_.erase(cit);
            hash_.erase(it);
            if (used_bitmap_)
                bitmap_[key] = false;
        }
    }

    void outdate(int key) {
        auto it = hash_.find(key);
        if (it != hash_.end()) {
            it->second.second = false;
            if (used_bitmap_)
                bitmap_[key] = false;
        }
    }

    vector<int> get_keys() {
        vector<int> keys;
        for (auto &iter : hash_) {
            // count valid keys
            if (iter.second.second)
                keys.push_back(iter.first);
        }
        sort(keys.begin(), keys.end());
        return keys;
    }
};
} // namespace laia_cache

#endif