#ifndef MINILRUCACHE_H
#define MINILRUCACHE_H

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
            unsigned int capacity_;

        public:
            // MiniLRUCache(unsigned int capacity) : capacity_(capacity) {}
            // MiniLRUCache(int capacity) : capacity_(capacity) {}
            MiniLRUCache() {}

            void set_cap(int capacity) {
                capacity_ = capacity;
            }

            bool check(const int key) const {
                auto it = hash_.find(key);
                if (it == hash_.end() || it->second.second == false) {
                    return false;
                }
                else {
                    return true;
                }
            }

            bool is_full() const {
                return hash_.size() == capacity_;
            }

            int get(int key) {
                auto it = hash_.find(key);
                if(it == hash_.end()) {
                    return insert(key);
                }
                else {
                    auto cit = it->second.first;
                    list_.erase(cit);
                    list_.push_front(key);
                    it->second = make_pair(list_.begin(), true);
                    return -1;
                }
            }

            // ensure that key does not exist in hash before insertion
            int insert(int key) {
                list_.push_front(key);
                hash_[key] = make_pair(list_.begin(), true);
                if(hash_.size() > capacity_) {
                    int ekey = list_.back();
                    hash_.erase(ekey);
                    list_.pop_back();
                    return ekey;
                }
                return -1;
            }

            void evict(int key) {
                auto it = hash_.find(key);
                if(it != hash_.end()) {
                    auto cit = it->second.first;
                    list_.erase(cit);
                    hash_.erase(it);
                }
            }

            void outdate(int key) {
                auto it = hash_.find(key);
                if(it != hash_.end()) {
                    it->second.second = false;
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
}

#endif