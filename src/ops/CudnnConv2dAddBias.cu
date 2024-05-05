#include "gpu_runtime.h"


__global__ void conv2d_add_bias(size_t nthreads,
    const float *input_data,
    float *output_data,
    size_t input_size,
    size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nthreads)
    return;
    size_t input_id = id % input_size / output_size;
    output_data[id] += input_data[input_id];
}


int Cudnn_Conv2dAddBias(const DLArrayHandle input_x, const DLArrayHandle input_f,
                      const DLArrayHandle bias, DLArrayHandle output,
                      const int padding_h, const int padding_w,
                      const int stride_h, const int stride_w,
                      DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    const float *input_data = (const float *)input_x->data;

    // input
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));
    size_t filter_N = input_f->shape[0];
    size_t filter_C = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    const float *filter_data = (const float *)input_f->data;

    // filter
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, filter_N, filter_C,
                                          filter_H, filter_W));

    // convolution
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    size_t out_N = output->shape[0];
    size_t out_C = output->shape[1];
    size_t out_H = output->shape[2];
    size_t out_W = output->shape[3];
    // output
    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, out_N, out_C, out_H,
                                          out_W));
    float *output_data = (float *)output->data;
    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    int fw_alg_cnt = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_map[dev_id],
                                                           &fw_alg_cnt));
    cudnnConvolutionFwdAlgoPerf_t* perf = new cudnnConvolutionFwdAlgoPerf_t[fw_alg_cnt];
    // algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    // tune this function to work in cudnn8
    int rtn_alg_cnt = 0;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
        cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc,
        fw_alg_cnt, &rtn_alg_cnt, perf));
    // use the fastest algorithm
    algo = perf[0].algo;
    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc, algo,
        &workspace_size));

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *work_data = find_chunk(workspace_size, dev_id);

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn_map[dev_id], &alpha, input_desc, input_data, filter_desc,
        filter_data, conv_desc, algo, work_data, workspace_size, &beta,
        out_desc, output_data));
    del_chunk(work_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    
    // add bias
    const float *bias_data = (const float*)bias->data;
    size_t nthreads = out_N * out_C * out_H * out_W;
    size_t BLOCKS = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t bias_output_size = out_H * out_W;
    size_t bias_input_size = out_C * bias_output_size;
    if (stream_handle)
        conv2d_add_bias<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, bias_data, output_data, bias_input_size, bias_output_size);
    else
        conv2d_add_bias<<<BLOCKS, THREADS_PER_BLOCK>>>(
            nthreads, bias_data, output_data, bias_input_size, bias_output_size);
    return 0;
}
