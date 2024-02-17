import hetu as ht
from hetu import init

import numpy as np
import time

def dfmslr_avazu(dense_input, sparse_input, y_, embeding_size=128):
    feature_dimension = 8930545
    embedding_size = embeding_size
    learning_rate = 0.001

    # sparse_feature_num = 18
    dense_feature_num = 4
    
    # FM
    bias = init.constant([1,1], fill_value=0.01, name="bias", ctx=ht.cpu(0))
    Embedding1 = init.random_normal(
        [feature_dimension, 1], stddev=0.01, name="fst_order_embedding", ctx=ht.cpu(0))
    FM_W = init.random_normal([dense_feature_num, 1], stddev=0.01, name="dense_parameter")
    sparse_1dim_input = ht.embedding_lookup_op(
        Embedding1, sparse_input, ctx=ht.cpu(0))
    fm_dense_part = ht.matmul_op(dense_input, FM_W)
    fm_sparse_part = ht.reduce_sum_op(sparse_1dim_input, axes=1)

    # fm_dense_part = ht.dropout_op(fm_dense_part, keep_prob = 1.0)
    # fm_sparse_part = ht.dropout_op(fm_sparse_part, keep_prob = 1.0)

    # fst order output
    y = fm_dense_part + fm_sparse_part + bias
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
