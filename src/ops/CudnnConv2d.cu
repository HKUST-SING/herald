#include "gpu_runtime.h"
#include <cudnn_cnn_infer.h>
#include <cudnn_cnn_train.h>

int CuDNN_DLGpuConv2d(const DLArrayHandle input_x, const DLArrayHandle input_f,
                      DLArrayHandle output, const int padding_h,
                      const int padding_w, const int stride_h,
                      const int stride_w, DLStreamHandle stream_handle = NULL) {
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
    delete[] perf;
    return 0;
}
int CuDNN_DLGpuConv2d_Gradient_of_Filter(const DLArrayHandle input_x,
                                         const DLArrayHandle gradient_y,
                                         DLArrayHandle gradient_f,
                                         const int padding_h,
                                         const int padding_w,
                                         const int stride_h, const int stride_w,
                                         DLStreamHandle stream_handle = NULL) {
    // create handle
    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    const float *input_data = (const float *)input_x->data;

    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));
    // dy
    size_t dy_N = gradient_y->shape[0];
    size_t dy_C = gradient_y->shape[1];
    size_t dy_H = gradient_y->shape[2];
    size_t dy_W = gradient_y->shape[3];
    const float *dy_data = (const float *)gradient_y->data;

    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));

    // conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // dw
    size_t df_N = gradient_f->shape[0];
    size_t df_C = gradient_f->shape[1];
    size_t df_H = gradient_f->shape[2];
    size_t df_W = gradient_f->shape[3];
    float *df_data = (float *)gradient_f->data;

    cudnnFilterDescriptor_t df_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&df_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        df_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, df_N, df_C, df_H, df_W));

    // algo
    cudnnConvolutionBwdFilterAlgo_t algo;
    int filter_alg_cnt = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_map[dev_id],
                                                           &filter_alg_cnt));
    cudnnConvolutionBwdFilterAlgoPerf_t *perf =
        new cudnnConvolutionBwdFilterAlgoPerf_t[filter_alg_cnt];
    int rtn_alg_cnt = 0;
    CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc,
        filter_alg_cnt, &rtn_alg_cnt, perf));
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    algo = perf[0].algo;
    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc, algo,
        &workspace_size));
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *work_data = find_chunk(workspace_size, dev_id);
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
        cudnn_map[dev_id], &alpha, input_desc, input_data, dy_desc, dy_data,
        conv_desc, algo, work_data, workspace_size, &beta, df_desc, df_data));
    del_chunk(work_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(df_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    delete[] perf;
    return 0;
}

int CuDNN_DLGpuConv2d_Gradient_of_Data(const DLArrayHandle input_f,
                                       const DLArrayHandle gradient_y,
                                       DLArrayHandle gradient_x,
                                       const int padding_h, const int padding_w,
                                       const int stride_h, const int stride_w,
                                       DLStreamHandle stream_handle = NULL) {
    // create handle
    int dev_id = (input_f->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // filter
    size_t filter_N = input_f->shape[0];
    size_t filter_C = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    const float *filter_data = (const float *)input_f->data;

    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, filter_N, filter_C,
                                          filter_H, filter_W));
    // dy
    size_t dy_N = gradient_y->shape[0];
    size_t dy_C = gradient_y->shape[1];
    size_t dy_H = gradient_y->shape[2];
    size_t dy_W = gradient_y->shape[3];
    const float *dy_data = (const float *)gradient_y->data;

    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));

    // conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // dx
    size_t dx_N = gradient_x->shape[0];
    size_t dx_C = gradient_x->shape[1];
    size_t dx_H = gradient_x->shape[2];
    size_t dx_W = gradient_x->shape[3];
    float *dx_data = (float *)gradient_x->data;

    cudnnTensorDescriptor_t dx_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dx_N, dx_C, dx_H, dx_W));

    // algo
    cudnnConvolutionBwdDataAlgo_t algo;
    int bw_alg_cnt = 0;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        cudnn_map[dev_id], &bw_alg_cnt));
    cudnnConvolutionBwdDataAlgoPerf_t* perf = new cudnnConvolutionBwdDataAlgoPerf_t[bw_alg_cnt];
    int rtn_alg_cnt = 0;
    CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc,
        bw_alg_cnt, &rtn_alg_cnt, perf));
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
    algo = perf[0].algo;
    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc, algo,
        &workspace_size));
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *work_data = find_chunk(workspace_size, dev_id);

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardData(
        cudnn_map[dev_id], &alpha, filter_desc, filter_data, dy_desc, dy_data,
        conv_desc, algo, work_data, workspace_size, &beta, dx_desc, dx_data));
    del_chunk(work_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    delete[] perf;
    return 0;
}
