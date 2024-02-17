#ifndef SHARE_MEM_H
#define SHARE_MEM_H

#include <ratio>
#include <sys/types.h>
#include <cstdint>
#include <memory>

#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "ring_buffer.h"

using namespace boost::interprocess;

/**
 * @brief a simple ring buffer implemented with shared memory
 *
 * This share memory only allows read & write 8-bytes data (uint64_t)
 * only one reader and one writer are allowed
 * Basic memory structure:
 * -----------------------
 * | data region         |
 * ------------------------
 * | read_ptr             |
 * | write_ptr            |
 * | other control info.. |
 * | total 400 bytes       |
 * -----------------------
 */
class SharedMemBuf {
public:
    static constexpr size_t defaultBuferSize = 1 << 20;
    static constexpr size_t defaultControlRegionSize = 400;
    SharedMemBuf(const std::string &name, const bool create = false,
                 const size_t size = defaultBuferSize,
                 boost::interprocess::mode_t mode = read_write) {
        name_ = name;
        create_ = create;
        uint64_t total_size;
        try {
            if (!create) {
                shm_ = shared_memory_object(open_only, name_.c_str(), mode);
                region_ = mapped_region(shm_, read_write);
                total_size = region_.get_size();
            } else {
                // create a new shared memory
                shm_ =
                    shared_memory_object(open_or_create, name_.c_str(), mode);
                data_region_size_ = size;
                // size should be power of 2
                if (not(size != 0 && (size & (size - 1)) == 0)) {
                    // round up to power of 2
                    int k = std::ceil(std::log2(size));
                    data_region_size_ = 1 << k;
                }
                total_size = data_region_size_ + control_region_byte_size_;
                shm_.truncate(total_size);
                region_ = mapped_region(shm_, read_write);
            }
        } catch (const boost::interprocess::interprocess_exception &ex) {
            std::cerr << "Failed to create or open shared memory object: "
                      << name_ << " Error: " << ex.what() << std::endl;
            throw;
        }
        buffer = reinterpret_cast<uint64_t *>(region_.get_address());
        data_buffer = buffer;
        data_region_size_ = total_size - control_region_byte_size_;
        control_start_ = data_region_size_ / sizeof(uint64_t);
        // these two address save the read & write pointer
        read_index_ = reinterpret_cast<uint64_t *>(&buffer[control_start_]);
        write_index_ =
            reinterpret_cast<uint64_t *>(&buffer[control_start_ + 1]);
        // setup data region
        if (create_) {
            *read_index_ = 0;
            *write_index_ = 0;
            // we need to construct the mutex and condition variable in place
            // with share memory
            mutex_ = new (&buffer[control_start_ + 2]) interprocess_mutex;
            cond_full_ =
                new (&buffer[control_start_ + 2
                             + sizeof(interprocess_mutex) / sizeof(uint64_t)])
                    interprocess_condition;
            cond_empty_ = new (
                &buffer[control_start_ + 2
                        + sizeof(interprocess_mutex) / sizeof(uint64_t)
                        + sizeof(interprocess_condition) / sizeof(uint64_t)])
                interprocess_condition;
        } else {
            // just init the pointer
            mutex_ = reinterpret_cast<interprocess_mutex *>(
                &buffer[control_start_ + 2]);
            cond_full_ = reinterpret_cast<interprocess_condition *>(
                &buffer[control_start_ + 2
                        + sizeof(interprocess_mutex) / sizeof(uint64_t)]);
            cond_empty_ = reinterpret_cast<interprocess_condition *>(
                &buffer[control_start_ + 2
                        + sizeof(interprocess_mutex) / sizeof(uint64_t)
                        + sizeof(interprocess_condition) / sizeof(uint64_t)]);
        }
        // setup ring buffer
        ring_buffer_ =
            new RingBuffer((void *)data_buffer, data_region_size_,
                           sizeof(uint64_t), read_index_, write_index_);
    }

    ~SharedMemBuf() {
        if (create_) {
            shared_memory_object::remove(name_.c_str());
        }
    }

    int send_data(const std::vector<uint64_t> &data) {
        // boost::interprocess::scoped_lock<interprocess_mutex> lock(*mutex_);
        // check if there is enough space in number of bytes
        uint64_t num_elem = data.size();
        int ret;
        if (ring_buffer_->unused_len() < num_elem + 1) {
            return -1;
        }
        ret = ring_buffer_->copy_in(&num_elem, 1);
        assert(ret == 1);
        ret = ring_buffer_->copy_in((void *)data.data(), data.size());
        assert(ret > 0 && size_t(ret) == num_elem);
        return num_elem;
    }

    std::vector<uint64_t> recv_data() {
        // boost::interprocess::scoped_lock<interprocess_mutex> lock(*mutex_);
        std::vector<uint64_t> data;
        data.clear();
        int ret;
        uint64_t num_elem = 0;
        ret = ring_buffer_->copy_out(&num_elem, 1);
        if (ret < 0) {
            return data;
        }
        data.resize(num_elem);
        size_t total_len = 0;
        while (1) {
            ret = ring_buffer_->copy_out(data.data() + total_len,
                                         num_elem - total_len);
            if (ret < 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            } else {
                total_len += ret;
                if (total_len == num_elem) {
                    break;
                }
            }
        }
        return data;
    }

    size_t queue_length() {
        return ring_buffer_->used_len();
    }

private:
    std::string name_;
    shared_memory_object shm_;
    mapped_region region_;
    bool create_;
    uint64_t *buffer;
    uint64_t *data_buffer;
    RingBuffer *ring_buffer_;

    // read & write pointers: index of the data region
    uint64_t *read_index_;
    uint64_t *write_index_;

    // control region index
    uint64_t control_start_;
    uint64_t control_region_byte_size_ = defaultControlRegionSize;
    // number of bytes
    uint64_t data_region_size_;

    // synchroization
    interprocess_mutex *mutex_;
    interprocess_condition *cond_full_;
    interprocess_condition *cond_empty_;
};

#endif // SHARE_MEM_H