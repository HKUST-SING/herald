#include "gpu_runtime.h"
#include <nccl.h>

int CuSparse_DLGpuCsrmv(const DLArrayHandle data_handle,
                        const DLArrayHandle row_handle,
                        const DLArrayHandle col_handle, int nrow, int ncol,
                        bool transpose, const DLArrayHandle input_handle,
                        DLArrayHandle output_handle,
                        DLStreamHandle stream_handle = NULL) {
    assert(data_handle->ndim == 1);
    assert(row_handle->ndim == 1);
    assert(col_handle->ndim == 1);
    assert(transpose ? nrow == input_handle->shape[0] :
                       ncol == input_handle->shape[0]);

    int nnz = data_handle->shape[0];
    int dev_id = (data_handle->ctx).device_id;
    cusp_init(dev_id, stream_handle);

    float alpha = 1.0;
    float beta = 0.0;

    cusparseMatDescr_t descr = 0;
    CUSP_CALL(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseOperation_t trans = transpose ? CUSPARSE_OPERATION_TRANSPOSE :
                                            CUSPARSE_OPERATION_NON_TRANSPOSE;
    // CUSP_CALL(cusparseScsrmv(
    //     cusp_map[dev_id], trans, nrow, ncol, nnz, (const float *)&alpha,
    //     descr, (const float *)data_handle->data, (const int
    //     *)row_handle->data, (const int *)col_handle->data, (const float
    //     *)input_handle->data, (const float *)&beta, (float
    //     *)output_handle->data));
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = nullptr;
    size_t buffer_size = 0;
    CUSP_CALL(cusparseCreateCsr(&matA, nrow, ncol, nnz, row_handle->data,
                      col_handle->data, data_handle->data, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CUSP_CALL(cusparseCreateDnVec(&vecX, ncol, input_handle->data, CUDA_R_32F));
    CUSP_CALL(cusparseCreateDnVec(&vecY, nrow, output_handle->data, CUDA_R_32F));

    cusparseSpMV_bufferSize(cusp_map[dev_id], trans, (const float*)&alpha, matA, vecX, (const float *)&beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer_size);
    CUSP_CALL(cusparseSpMV(cusp_map[dev_id], CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                           CUSPARSE_MV_ALG_DEFAULT, dBuffer));

    CUSP_CALL(cusparseDestroySpMat(matA));
    CUSP_CALL(cusparseDestroyDnVec(vecX));
    CUSP_CALL(cusparseDestroyDnVec(vecY));

    return 0;
}