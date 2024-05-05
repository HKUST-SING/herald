#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include <vector>
#include <stdint.h>
using namespace std;

using elem_t = uint64_t;

class Array {
public:
    Array();
    Array(size_t ndim, const vector<size_t> &shape, const elem_t &init_val = 0);
    Array(Array&& array);
    Array(const Array &array) = delete;
    ~Array();
    /* Array(Array &&array) = delete; */

    void reset(const elem_t &init_val);
    elem_t &operator()(int i);
    Array& operator=(Array&& array);
    /* virtual elem_t operator() (int index, ...); */

protected:
    size_t ndim_;
    vector<size_t> shape_;
    uint64_t num_elements;
    elem_t *buffer_;
};

class Array2D : public Array {
public:
    Array2D();
    Array2D(Array2D&& array);
    Array2D(const Array2D& array) = delete;
    Array2D(size_t ndim, const vector<size_t> &shape,
            const elem_t &init_val = 0);
    Array2D& operator=(Array2D&& array);
    elem_t *operator[](const size_t &i);
    elem_t &operator()(const size_t &i, const size_t &j);
};

class Array3D : public Array {
public:
    Array3D();
    Array3D(Array3D &&array);
    Array3D(const Array3D& array) = delete;
    Array3D(size_t ndim, const vector<size_t> &shape,
            const elem_t &init_val = 0);
    Array3D& operator=(Array3D&& array);
    elem_t &operator()(const size_t &i, const size_t &j, const size_t &z);
};

#endif /* ARRAY_H */