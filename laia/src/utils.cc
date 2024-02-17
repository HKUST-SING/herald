#include "utils.h"
#include <complex>
#include <cstddef>
#include <cstring>
#include <pybind11/attr.h>

namespace laia_cache {
RawArray2D init_2d_array(int dim1, int dim2, elem_t init_val) {
    RawArray2D array;
    array = new elem_t *[dim1];
    for (int i = 0; i < dim1; i++) {
        array[i] = new elem_t[dim2];
        for (int j = 0; j < dim2; j++)
            array[i][j] = init_val;
    }
    return array;
}

RawArray3D init_3d_array(int dim1, int dim2, int dim3, elem_t init_val) {
    RawArray3D array;
    array = new RawArray2D[dim1];
    for (int i = 0; i < dim1; i++) {
        array[i] = init_2d_array(dim2, dim3, init_val);
    }
    return array;
}

void release_2d_array(RawArray2D array, int dim1,
                      __attribute__((unused)) int dim2) {
    if (array == nullptr)
        return;
    for (int i = 0; i < dim1; i++) {
        if (array[i])
            delete[] array[i];
    }
    delete[] array;
}

void release_3d_array(RawArray3D array, int dim1, int dim2, int dim3) {
    if (array == nullptr)
        return;
    for (int i = 0; i < dim1; i++) {
        release_2d_array(array[i], dim2, dim3);
    }
    delete[] array;
}

void reset_2d_array(RawArray2D array, int dim1, int dim2, elem_t init_val) {
    if (array == nullptr)
        return;
    for (int i = 0; i < dim1; i++) {
        if (array[i] != nullptr)
            memset(array[i], init_val, dim2 * sizeof(elem_t));
    }
}

void reset_3d_array(RawArray3D array, int dim1, int dim2, int dim3, elem_t init_val) {
    if (array == nullptr)
        return;
    for (int i = 0; i < dim1; i++) {
        reset_2d_array(array[i], dim2, dim3, init_val);
    }
}

} // namespace laia_cache
