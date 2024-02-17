#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <ctime>
#include <chrono>
#include <vector>
#include <boost/container/flat_set.hpp>

#include "array.h"

using namespace std;

namespace laia_cache {
#if defined(DEBUG)
#define TIMING(function)                                                       \
    {                                                                          \
        std::chrono::steady_clock::time_point begin =                          \
            std::chrono::steady_clock::now();                                  \
        do {                                                                   \
            function;                                                          \
        } while (0);                                                           \
        std::chrono::steady_clock::time_point end =                            \
            std::chrono::steady_clock::now();                                  \
        printf(                                                                \
            "Elapsed: %lu [us]\n",                                             \
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin) \
                .count());                                                     \
    }

#else
#define TIMING(a) a;
#endif

using elem_t = uint64_t;
using RawArray2D = elem_t **;
using RawArray3D = elem_t ***;
RawArray2D init_2d_array(int dim1, int dim2, elem_t init_val = 0);
RawArray3D init_3d_array(int dim1, int dim2, int dim3, elem_t init_val = 0);
void release_2d_array(RawArray2D array, int dim1, int dim2);
void release_3d_array(RawArray3D array, int dim1, int dim2, int dim3);

void reset_2d_array(RawArray2D array, int dim1, int dim2, elem_t init_val = 0);
void reset_3d_array(RawArray3D array, int dim1, int dim2, int dim3,
                    elem_t init_val = 0);

typedef uint64_t emb_key_t;
using dist_t = Array2D;
using comm_plan_t = vector<boost::container::flat_set<emb_key_t>>;
using element_t = vector<emb_key_t>;

} // namespace laia_cache

#endif /* UTILS_H */