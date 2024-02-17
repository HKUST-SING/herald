#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "array.h"

Array::Array() : ndim_(0), num_elements(0), buffer_(nullptr) {
}

Array::Array(size_t ndim, const vector<size_t> &shape, const elem_t &init_val) {
    ndim_ = ndim;
    assert(shape.size() == static_cast<unsigned long>(ndim_));
    shape_ = shape;
    num_elements = 1;
    for (auto &val : shape_) {
        num_elements *= val;
    }
    buffer_ = new elem_t[num_elements];
    reset(init_val);
}

Array::Array(Array &&array) :
    ndim_(array.ndim_), shape_(std::move(array.shape_)),
    num_elements(array.num_elements), buffer_(std::move(array.buffer_)) {
    array.buffer_ = nullptr;
}

Array::~Array() {
    if (buffer_)
        delete[] buffer_;
}

Array &Array::operator=(Array &&array) {
    if (&array != this) {
        ndim_ = std::move(array.ndim_);
        shape_ = std::move(array.shape_);
        num_elements = std::move(array.num_elements);
        if (buffer_)
            delete[] buffer_;
        buffer_ = std::move(array.buffer_);
        array.buffer_ = nullptr;
    }
    return *this;
}

void Array::reset(const elem_t &init_val) {
    if (init_val == 0) [[likely]] {
        /* memset should be much faster than std::fill for zero case */
        memset(buffer_, init_val, num_elements * sizeof(elem_t));
    } else
        std::fill(buffer_, buffer_ + num_elements, init_val);
}

Array2D::Array2D() : Array() {
}

Array2D::Array2D(size_t ndim, const vector<size_t> &shape,
                 const elem_t &init_val) :
    Array(ndim, shape, init_val) {
}

Array2D::Array2D(Array2D &&array) : Array(std::move(array)) {
}

elem_t *Array2D::operator[](const size_t &i) {
    if (not(i < shape_[0]))
        throw runtime_error("no such index: " + to_string(i));
    return buffer_ + i * shape_[1];
}

elem_t &Array2D::operator()(const size_t &i, const size_t &j) {
    assert(shape_.size() == 2);
    if (not(i < shape_[0] && j < shape_[1]))
        throw runtime_error("out of range: " + to_string(i) + ", "
                            + to_string(j));
    return *(buffer_ + i * shape_[1] + j);
}

Array2D &Array2D::operator=(Array2D &&array) {
    Array::operator=(std::move(array));
    return *this;
}

Array3D::Array3D() : Array() {
}

Array3D::Array3D(Array3D &&array) : Array(std::move(array)) {
}

Array3D::Array3D(size_t ndim, const vector<size_t> &shape,
                 const elem_t &init_val) :
    Array(ndim, shape, init_val) {
}

elem_t &Array3D::operator()(const size_t &i, const size_t &j, const size_t &z) {
    assert(shape_.size() == 3);
    if (not(i < shape_[0] and j < shape_[1] and z < shape_[2])) {
        throw runtime_error("out of range: " + to_string(i) + ", "
                            + to_string(j) + ", " + to_string(z));
    }
    return *(buffer_ + i * (shape_[1] * shape_[2]) + j * shape_[2] + z);
}

Array3D &Array3D::operator=(Array3D &&array) {
    Array::operator=(std::move(array));
    return *this;
}