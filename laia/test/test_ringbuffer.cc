#include <gtest/gtest.h>
#include "ring_buffer.h"
#include <thread>

class RingBufferTest : public ::testing::Test {
protected:
    RingBufferTest() :
        buf_(buffer, 4096 * sizeof(uint64_t), sizeof(uint64_t), &in, &out) {
        in = 0;
        out = 0;
    } // Assuming buffer size is 4096

    uint64_t buffer[4096];
    uint64_t in, out;
    RingBuffer buf_;
};

TEST_F(RingBufferTest, CopyInAndCopyOut) {
    uint64_t value_in = 123;
    ASSERT_EQ(buf_.copy_in(&value_in, 1), 1);

    uint64_t value_out;
    ASSERT_EQ(buf_.copy_out(&value_out, 1), 1);
    ASSERT_EQ(value_in, value_out);
}

TEST_F(RingBufferTest, CopyInAndCopyOutBoundary) {
    size_t num_iters = 4096 / sizeof(uint64_t);
    for (uint64_t i = 0; i < num_iters; ++i) {
        ASSERT_EQ(buf_.copy_in(&i, 1), 1);
    }

    for (uint64_t i = 0; i < num_iters; ++i) {
        uint64_t value_out;
        ASSERT_EQ(buf_.copy_out(&value_out, 1), 1);
        ASSERT_EQ(i, value_out);
    }
}

TEST_F(RingBufferTest, ProducerConsumer) {
    std::thread producer([this]() {
        for (uint64_t i = 0; i < 4096; ++i) {
            while (buf_.copy_in(&i, 1) < 0) {
                // Buffer is full, wait and retry
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    std::thread consumer([this]() {
        for (uint64_t i = 0; i < 4096; ++i) {
            uint64_t value_out;
            while (buf_.copy_out(&value_out, 1) < 0) {
                // Buffer is empty, wait and retry
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ASSERT_EQ(i, value_out);
        }
    });

    producer.join();
    consumer.join();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
