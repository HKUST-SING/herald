#include "gpu_runtime.h"

// cuda init for set the cuda device
void cuda_init() {
    int devicedChoosed;
    cudaGetDevice(&devicedChoosed);
    cudaSetDevice(devicedChoosed);
}

// check whether the cudnn is init
// if not : init a cudnn handle
// else: do nothing
std::map<size_t, bool> is_cudnn_init;
std::map<size_t, cudnnHandle_t> cudnn_map;
void cudnn_init(size_t dev_id, DLStreamHandle stream) {
    if (is_cudnn_init.find(dev_id) == is_cudnn_init.end()) {
        is_cudnn_init.insert(std::pair<size_t, bool>(dev_id, false));
        cudnnHandle_t cudnn;
        cudnn_map.insert(std::pair<size_t, cudnnHandle_t>(dev_id, cudnn));
        CUDNN_CALL(cudnnCreate(&cudnn_map[dev_id]));
        is_cudnn_init[dev_id] = true;
    }
    if (stream) {
        CUDNN_CALL(
            cudnnSetStream(cudnn_map[dev_id], *(cudaStream_t *)stream->handle));
    }
}

std::map<size_t, bool> is_cusp_init;
std::map<size_t, cusparseHandle_t> cusp_map;
void cusp_init(size_t dev_id, DLStreamHandle stream) {
    if (is_cusp_init.find(dev_id) == is_cusp_init.end()) {
        is_cusp_init.insert(std::pair<size_t, bool>(dev_id, false));
        cusparseHandle_t cusp;
        cusp_map.insert(std::pair<size_t, cusparseHandle_t>(dev_id, cusp));
        CUSP_CALL(cusparseCreate(&cusp_map[dev_id]));
        is_cusp_init[dev_id] = true;
    }
    if (stream) {
        CUSP_CALL(cusparseSetStream(cusp_map[dev_id],
                                    *(cudaStream_t *)stream->handle));
    }
}

std::map<size_t, bool> is_cublas_init;
std::map<size_t, cublasHandle_t> cublas_map;
void cublas_init(size_t dev_id, DLStreamHandle stream) {
    if (is_cublas_init.find(dev_id) == is_cublas_init.end()) {
        is_cublas_init.insert(std::pair<size_t, bool>(dev_id, false));
        cublasHandle_t cusp;
        cublas_map.insert(std::pair<size_t, cublasHandle_t>(dev_id, cusp));
        CUBLAS_CALL(cublasCreate(&cublas_map[dev_id]));
        is_cublas_init[dev_id] = true;
    }
    if (stream) {
        CUBLAS_CALL(cublasSetStream(cublas_map[dev_id],
                                    *(cudaStream_t *)stream->handle));
    }
}

int DLStreamCreate(size_t dev_id, DLStreamHandle *handle) {
    DLStream *s = new DLStream();
    s->device_id = dev_id;
    s->handle = nullptr;
    CUDA_CALL(cudaSetDevice(dev_id));
    cudaStream_t *stream_handle = new cudaStream_t();
    CUDA_CALL(cudaStreamCreate(stream_handle));
    s->handle = stream_handle;
    *handle = s;
    // delete s;
    return 0;
}

int DLStreamDestroy(DLStreamHandle handle) {
    CUDA_CALL(cudaStreamDestroy(*(cudaStream_t *)handle->handle));
    return 0;
}

int DLStreamSync(DLStreamHandle handle) {
    cudaStreamSynchronize(*(cudaStream_t *)handle->handle);
    return 0;
}

int DLEventCreate(size_t dev_id, DLEventHandle *handle) {
    DLEvent *s = new DLEvent();
    s->device_id = dev_id;
    s->handle = nullptr;
    CUDA_CALL(cudaSetDevice(dev_id));
    cudaEvent_t *event_handle = new cudaEvent_t();
    CUDA_CALL(cudaEventCreate(event_handle));
    s->handle = event_handle;
    *handle = s;
    // delete s;
    return 0;
}

int DLEventDestroy(DLEventHandle handle) {
    CUDA_CALL(cudaEventDestroy(*(cudaEvent_t *)handle->handle));
    return 0;
}

int DLEventRecord(DLStreamHandle stream_handle, DLEventHandle event_handle) {
    CUDA_CALL(cudaEventRecord(*(cudaEvent_t *)event_handle->handle,
                              *(cudaStream_t *)stream_handle->handle));
    return 0;
}

int DLEventSync(DLEventHandle handle) {
    CUDA_CALL(cudaEventSynchronize(*(cudaEvent_t *)handle->handle));
    return 0;
}

int DLEventElapsedTime(DLEventHandle start, DLEventHandle ending,
                       float *duration) {
    CUDA_CALL(cudaEventElapsedTime(duration, *(cudaEvent_t *)start->handle,
                                   *(cudaEvent_t *)ending->handle));
    return 0;
}
