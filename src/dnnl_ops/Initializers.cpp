#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <omp.h>
#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"

int cpu_NormalInit(DLArrayHandle arr, const float mean, const float stddev,
                   unsigned long long seed) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    size_t n_threads = (size >> 25) + 1;
    if (n_threads > 16)
        n_threads = 16;

    std::normal_distribution<float> normal_dist(mean, stddev);
#pragma omp parallel num_threads(n_threads)
    {
        size_t rank = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();
        std::default_random_engine generator(seed + rank);
        size_t length = size / num_threads;
        size_t start = rank * length;
        size_t ending = start + length;
        if (rank == num_threads - 1)
            ending = size;
        for (size_t j = start; j < ending; ++j) {
            arr_data[j] = normal_dist(generator);
        }
    }

    return 0;
}

int cpu_UniformInit(DLArrayHandle arr, const float lb, const float ub,
                    unsigned long long seed) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    size_t n_threads = (size >> 25) + 1;
    if (n_threads > 16)
        n_threads = 16;

    std::uniform_real_distribution<float> uniform_dist(lb, ub);
#pragma omp parallel num_threads(n_threads)
    {
        size_t rank = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();
        std::default_random_engine generator(seed + rank);
        size_t length = size / num_threads;
        size_t start = rank * length;
        size_t ending = start + length;
        if (rank == num_threads - 1)
            ending = size;
        for (size_t j = start; j < ending; ++j) {
            arr_data[j] = uniform_dist(generator);
        }
    }

    return 0;
}

int cpu_TruncatedNormalInit(DLArrayHandle arr, const float mean,
                            const float stddev, unsigned long long seed) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    size_t n_threads = (size >> 25) + 1;
    if (n_threads > 16)
        n_threads = 16;

    std::normal_distribution<float> truncated_normal_dist(mean, stddev);
    float upper_limit = mean + 2 * stddev;
    float lower_limit = mean - 2 * stddev;
#pragma omp parallel num_threads(n_threads)
    {
        size_t rank = omp_get_thread_num();
        size_t num_threads = omp_get_num_threads();
        std::default_random_engine generator(seed + rank);
        size_t length = size / num_threads;
        size_t start = rank * length;
        size_t ending = start + length;
        if (rank == num_threads - 1)
            ending = size;
        for (size_t j = start; j < ending; ++j) {
            float temp = truncated_normal_dist(generator);
            while (temp > upper_limit || temp < lower_limit)
                temp = truncated_normal_dist(generator);
            arr_data[j] = temp;
        }
    }
    return 0;
}