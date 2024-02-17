#include <cstdint>
#include <gtest/gtest.h>
#include "share_mem.h"

class SharedMemBufTest : public ::testing::Test {
protected:
    SharedMemBufTest() :
        buf_create_("test_buf", true), buf_open_("test_buf", false) {
    }

    SharedMemBuf buf_create_;
    SharedMemBuf buf_open_;
};

TEST_F(SharedMemBufTest, SendAndRecvData) {
    std::vector<uint64_t> data_send = {1, 2, 3, 4, 5};
    std::cout << "data_send.size() = " << data_send.size() << "\n";
    ASSERT_EQ(buf_create_.send_data(data_send), data_send.size());

    std::vector<uint64_t> data_recv = buf_open_.recv_data();
    ASSERT_EQ(data_recv, data_send);
}

TEST_F(SharedMemBufTest, SendAndRecvDataBoundary) {
    // One slot is reserved for the size of the data
    std::vector<uint64_t> data_send(
        SharedMemBuf::defaultBuferSize / sizeof(uint64_t) - 1, 123);
    ASSERT_EQ(buf_create_.send_data(data_send), data_send.size());

    std::vector<uint64_t> data_recv = buf_open_.recv_data();
    ASSERT_EQ(data_recv, data_send);
}

TEST_F(SharedMemBufTest, MultiThreadSendAndRecvData) {
    std::vector<uint64_t> data_send = {1, 2, 3, 4, 5};

    std::thread receiver([&]() {
        for (int i = 0; i < 1000; ++i) {
            while (1) {
                std::vector<uint64_t> data_recv = buf_open_.recv_data();
                if (data_recv.size() > 0) {
                    ASSERT_EQ(data_recv, data_send);
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::thread sender([&]() {
        for (int i = 0; i < 1000; ++i) {
            while (1) {
                int ret = buf_create_.send_data(data_send);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (ret > 0)
                    break;
            }
        }
    });

    sender.join();
    receiver.join();
}

TEST_F(SharedMemBufTest, MultiProcessSendAndRecvData) {
    std::vector<uint64_t> data_send = {1, 2, 3, 4, 5};

    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        for (int i = 0; i < 1000; ++i) {
            while (1) {
                int ret = buf_create_.send_data(data_send);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (ret > 0)
                    break;
            }
        }
        exit(0);
    } else if (pid > 0) {
        // Parent process
        for (int i = 0; i < 1000; ++i) {
            while (1) {
                std::vector<uint64_t> data_recv = buf_open_.recv_data();
                if (data_recv.size() > 0) {
                    ASSERT_EQ(data_recv, data_send);
                    break;
                }
            }
        }
        wait(NULL); // Wait for child process to finish
    } else {
        // fork failed
        ASSERT_TRUE(false) << "Fork failed!";
    }
}

class SharedMemBufTestRaw : public ::testing::Test {};

TEST_F(SharedMemBufTestRaw, MultiProcessSendAndRecvData) {
    std::vector<uint64_t> data_send = {1, 2, 3, 4, 5};

    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        SharedMemBuf buf_create("test_buf", true);
        for (int i = 0; i < 1000; ++i) {
            auto data = data_send;
            for (auto &j : data)
                j *= i;
            while (1) {
                int ret = buf_create.send_data(data);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (ret > 0)
                    break;
            }
        }
        exit(0);
    } else if (pid > 0) {
        // Parent process
        usleep(10000); // Sleep for 10 milliseconds to ensure child process has
                       // created the shared memory
        SharedMemBuf buf_open("test_buf", false);
        for (int i = 0; i < 1000; ++i) {
            auto data = data_send;
            for (auto &j : data)
                j *= i;
            while (1) {
                std::vector<uint64_t> data_recv = buf_open.recv_data();
                if (data_recv.size() > 0) {
                    ASSERT_EQ(data_recv, data);
                    break;
                }
            }
        }
        wait(NULL); // Wait for child process to finish
    } else {
        // fork failed
        ASSERT_TRUE(false) << "Fork failed!";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
