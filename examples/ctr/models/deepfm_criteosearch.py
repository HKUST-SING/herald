import hetu as ht
from hetu import init

import numpy as np
import time


def dfm_criteosearch(dense_input, sparse_input, y_, embeding_size=128, enable_push_index=False):
    feature_dimension = 14859910
    embedding_size = embeding_size
    learning_rate = 0.001

    dense_feature_num = 3
    sparse_feature_num = 17

    # FM
    Embedding1 = init.random_normal(
        [feature_dimension, 1], stddev=0.001, name="fst_order_embedding", ctx=ht.cpu(0))
    FM_W = init.random_normal([dense_feature_num, 1], stddev=0.001, name="dense_parameter")
    sparse_1dim_input = ht.embedding_lookup_op(
        Embedding1, sparse_input, ctx=ht.cpu(0), enable_push_index=enable_push_index)
    fm_dense_part = ht.matmul_op(dense_input, FM_W)
    fm_sparse_part = ht.reduce_sum_op(sparse_1dim_input, axes=1)
    # fst order output
    y1 = fm_dense_part + fm_sparse_part

    Embedding2 = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.001, name="snd_order_embedding", ctx=ht.cpu(0))
    sparse_2dim_input = ht.embedding_lookup_op(
        Embedding2, sparse_input, ctx=ht.cpu(0))
    sparse_2dim_sum = ht.reduce_sum_op(sparse_2dim_input, axes=1)
    sparse_2dim_sum_square = ht.mul_op(sparse_2dim_sum, sparse_2dim_sum)

    sparse_2dim_square = ht.mul_op(sparse_2dim_input, sparse_2dim_input)
    sparse_2dim_square_sum = ht.reduce_sum_op(sparse_2dim_square, axes=1)
    sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
    sparse_2dim_half = sparse_2dim * 0.5
    # snd order output
    y2 = ht.reduce_sum_op(sparse_2dim_half, axes=1, keepdims=True)

    # DNN
    flatten = ht.array_reshape_op(sparse_2dim_input, (-1, sparse_feature_num * embedding_size))
    W1 = init.random_normal([sparse_feature_num * embedding_size, 256], stddev=0.001, name="W1")
    W2 = init.random_normal([256, 256], stddev=0.001, name="W2")
    W3 = init.random_normal([256, 1], stddev=0.001, name="W3")

    fc1 = ht.matmul_op(flatten, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    y3 = ht.matmul_op(relu2, W3)

    y4 = y1 + y2
    y = y4 + y3
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
