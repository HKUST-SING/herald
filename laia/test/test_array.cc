#include "binding.h"
#include "utils.h"
#include <iostream>
#include <iterator>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <queue>
#include <stdexcept>
#include <variant>

using namespace std;
using namespace laia_cache;

RawArray2D embeddings;
// Array2D results;
using dist_t = py::array_t<elem_t>;
using comm_plan_t = vector<vector<elem_t>>;
using queue_element_t = std::variant<comm_plan_t, dist_t, vector<elem_t>>;
queue<queue_element_t> results;

void recv_py_array(py::array_t<elem_t> embeds) {
    if (embeds.ndim() != 2) {
        cerr << "input should be 2-D numpy array" << endl;
        return;
    }
    // copy sample_embs_py to 2Darray
    int num_sample = embeds.shape()[0];
    int num_table = embeds.shape()[1];

    cout << "Input shape: " << num_sample << "; " << num_table << endl;
    embeddings = init_2d_array(num_sample, num_table);
    for (int i = 0; i < num_sample; i++) {
        for (int j = 0; j < num_table; j++) {
            embeddings[i][j] = embeds.at(i,j);
        }
    }

    // do some stuff

    // caculate results
    py::array_t<elem_t> output;
    int num_worker = 6;
    int batch_size = 4;
    output.resize({num_worker, batch_size});
    for (int i = 0; i < num_worker; i++) {
        for (int j = 0; j < batch_size; j++) {
            output.mutable_at(i, j) = i * num_table + j;
        }
    }
    results.emplace(output);

    auto ptr = output.data(3);
    cout << "pointer data: " << *ptr << endl;
    vector<elem_t> output_array(ptr, ptr + batch_size);
    results.emplace(output_array);
    // py::array_t<elem_t> other;
    // other.resize({3,8});
    // results.emplace(other);
    comm_plan_t some;
    for (int i = 0; i < num_worker; i++) {
        vector<elem_t> a;
        for (int j = 0; j < i+1; j++) {
            a.emplace_back(j);
        }
        some.emplace_back(a);
    }
    results.emplace(some);
}

queue_element_t get_dist() {
    if (not results.empty()) {
        auto res = results.front();
        results.pop();
        return res;
    } else {
        throw std::runtime_error("Results queue is empty");
    }
}

PYBIND11_MODULE(laia_test, m) {
    m.doc() = "laia test"; // optional module docstring

    m.def("recv_py_array", &recv_py_array);
    m.def("get_dist", &get_dist);
} // PYBIND11_MODULE
