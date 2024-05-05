#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <sys/time.h>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"

using namespace dnnl;
using namespace std;

extern "C" int DnnlMatrixElementwiseAddByConst(const DLArrayHandle input,
                                               float val,
                                               DLArrayHandle output) {
    // engine eng(engine::kind::cpu, 0);
    // stream engine_stream(eng);
    dnnl_stream_init();

    vector<long int> shape, format;
    for (int i = 0; i < input->ndim; i++)
        shape.push_back(input->shape[i]);
    format.resize(input->ndim);
    format[(input->ndim) - 1] = 1;
    for (int i = format.size() - 2; i >= 0; i--)
        format[i] = format[i + 1] * shape[i + 1];

    auto mat_md = memory::desc(shape, memory::data_type::f32, format);
    auto src_mem = memory(mat_md, eng, input->data);
    auto dst_mem = memory(mat_md, eng, output->data);

    auto AddConst_d =
        eltwise_forward::desc(prop_kind::forward_training,
                              algorithm::eltwise_linear, mat_md, 1.f, val);
    auto AddConst_pd = eltwise_forward::primitive_desc(AddConst_d, eng);
    auto AddConst = eltwise_forward(AddConst_pd);

    AddConst.execute(engine_stream,
                     {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();
    return 0;
}
