import hetu as ht
from hetu import init

import numpy as np
import time


def dfmsfm_criteosearch(dense_input, sparse_input, y_, embeding_size=128, enable_push_index=False):
    feature_dimension = 14859910
    # [12, 9, 17982, 18, 54211, 22, 173, 1033, 1551, 836, 213, 6, 24, 1542350, 756064, 311, 12485095, 0]
    # 14859910 criteo_search
    embedding_size = embeding_size
    learning_rate = 0.001
    # sparse_feature_num = 17
    dense_feature_num = 3

    # FM
    bias = init.constant([1,1], fill_value=0.01, name="bias", ctx=ht.cpu(0))
    Embedding1 = init.random_normal(
        [feature_dimension, 1], stddev=0.01, name="fst_order_embedding", ctx=ht.cpu(0))
    FM_W = init.random_normal([dense_feature_num, 1], stddev=0.01, name="dense_parameter")
    sparse_1dim_input = ht.embedding_lookup_op(
        Embedding1, sparse_input, ctx=ht.cpu(0), enable_push_index=enable_push_index)
    fm_dense_part = ht.matmul_op(dense_input, FM_W)
    fm_sparse_part = ht.reduce_sum_op(sparse_1dim_input, axes=1)
    # fst order output
    y1 = fm_dense_part + fm_sparse_part

    Embedding2 = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding", ctx=ht.cpu(0))
    sparse_2dim_input = ht.embedding_lookup_op(
        Embedding2, sparse_input, ctx=ht.cpu(0), enable_push_index=enable_push_index)
    sparse_2dim_sum = ht.reduce_sum_op(sparse_2dim_input, axes=1)
    sparse_2dim_sum_square = ht.mul_op(sparse_2dim_sum, sparse_2dim_sum)

    sparse_2dim_square = ht.mul_op(sparse_2dim_input, sparse_2dim_input)
    sparse_2dim_square_sum = ht.reduce_sum_op(sparse_2dim_square, axes=1)
    sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
    sparse_2dim_half = sparse_2dim * 0.5
    # snd order output
    y2 = ht.reduce_sum_op(sparse_2dim_half, axes=1, keepdims=True)

    y = y1 + y2 + bias

    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate) # SGDOptimizer AdamOptimizer
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
