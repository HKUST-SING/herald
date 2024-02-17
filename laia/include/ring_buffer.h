#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>

// #define barrier() __asm__ __volatile__("": : :"memory")
// #define barrier() __asm__ volatile("sfence" ::: "memory")
#define barrier() __asm__ __volatile__("": : :"memory")

class RingBuffer {
public:
    /**
     * @brief Construct a new Ring Buffer object
     * @param buffer the shared memory buffer
     * @param size the size of the buffer in bytes, should be power of 2
     * @param esize the size of each element in bytes, eg, 8 bytes for uint64_t
     */
    RingBuffer(void *buffer, size_t size, size_t esize, uint64_t *in_ptr,
               uint64_t *out_ptr) {
        data_ = buffer;
        size /= esize;
        size_ = size;
        mask_ = size - 1;
        in_ = in_ptr;
        out_ = out_ptr;
        esize_ = esize;
    }

    /**
     * @brief Copy elements to the ring buffer
     * @param src the source buffer
     * @param len the number of elements to copy
     * @return the number of elements copied or -1 if not enough space
     */
    int copy_in(void *src, size_t len) {
        auto l = unused_len();
        if (len > l) {
            return -1;
        }

        auto offset = *in_;
        copy_in_helper(src, len, offset);
        *in_ += len;
        return len;
    }

    /**
     * @brief Copy elements from the ring buffer
     * @param buf the destination buffer
     * @param len the number of elements to copy
     * @return the number of elements copied or -1 if not enough data
     */

    int copy_out(void *buf, size_t len) {
        auto l = *in_ - *out_;
        if (len > l) {
            return -1;
        }

        auto offset = *out_;
        copy_out_helper(buf, len, offset);
        *out_ += len;
        return len;
    }

    bool empty() {
        return *in_ == *out_;
    }

    bool full() {
        return unused_len() == 0;
    }

    inline size_t unused_len() {
        return (mask_ + 1) - (*in_ - *out_);
    }

    inline size_t used_len() {
        return *in_ - *out_;
    } 

private:
    void copy_in_helper(void *src, size_t len, uint64_t offset) {
        size_t l;

        offset &= mask_;
        if (esize_ != 1) {
            offset *= esize_;
            size_ *= esize_;
            len *= esize_;
        }

        l = std::min(len, size_ - offset);
        std::memcpy(data_ + offset, src, l);
        std::memcpy(data_, src + l, len - l);
        barrier();
    }

    void copy_out_helper(void *dst, size_t len, uint64_t offset) {
        size_t l;

        if (esize_ != 1) {
            offset *= esize_;
            size_ *= esize_;
            len *= esize_;
        }

        l = std::min(len, size_ - offset);
        std::memcpy(dst, data_ + offset, l);
        std::memcpy(dst + l, data_, len - l);
        barrier();
    }

private:
    void *data_;
    // size - 1
    size_t mask_;
    size_t size_;
    size_t esize_;
    uint64_t *in_;
    uint64_t *out_;
};

#endif // RING_BUFFER_H