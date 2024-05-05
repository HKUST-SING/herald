#include "thread_pool.h"
#include <mutex>
#include <iostream>

static ThreadPool *pool;
const size_t kThreadNum = 16;

ThreadPool::ThreadPool(size_t thread_num) :
    terminate_(false), thread_num_(thread_num), complete_task_num_(0) {
    for (size_t i = 0; i < thread_num; ++i) {
        threads_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mutex_);
                    this->cond_.wait(lock, [this] {
                        return this->terminate_ || !this->tasks_.empty();
                    });

                    if (this->terminate_ && this->tasks_.empty())
                        return;

                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                try {
                    task();
                } catch (std::exception &e) {
                    std::cerr << "running task, with exception..." << e.what()
                              << std::endl;
                    return;
                }
                complete_task_num_++;
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        terminate_ = true;
    }
    cond_.notify_all();

    for (std::thread &thread : threads_) {
        thread.join();
    }
}

void ThreadPool::Wait(int task_num) {
    while (complete_task_num_ != task_num) {
        usleep(100);
    }
    complete_task_num_ = 0;
}

bool ThreadPool::empty() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    return tasks_.empty();
}

void ThreadPool::clear() {
    std::unique_lock<std::mutex> lock(this->mutex_);
    std::queue<std::function<void()>> empty_queue;
    std::swap(tasks_, empty_queue);
}

ThreadPool *ThreadPool::Get() {
    if (!pool) {
        pool = new ThreadPool(kThreadNum);
    }
    return pool;
}
