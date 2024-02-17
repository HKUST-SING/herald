import hetu as ht
from hetu import init

import numpy as np
import time

num_hot_emb = 162056

def cross_layer(x0, x1, sparse_feature_num, dense_feature_num, embedding_size=128):
    # x0: input embedding feature (batch_size, 18 * embedding_size + 4)
    # x1: the output of last layer (batch_size, 18 * embedding_size + 4)

    embedding_len = sparse_feature_num * embedding_size + dense_feature_num
    weight = init.random_normal(
        shape=(embedding_len, 1), stddev=0.001, name='weight')
    bias = init.random_normal(shape=(embedding_len,), stddev=0.001, name='bias')
    x1w = ht.matmul_op(x1, weight)  # (batch_size, 1)
    y = ht.mul_op(x0, ht.broadcastto_op(x1w, x0))
    y = y + x1 + ht.broadcastto_op(bias, y)
    return y


def build_cross_layer(x0, sparse_feature_num, dense_feature_num, embedding_size, num_layers=3):
    x1 = x0
    for i in range(num_layers):
        x1 = cross_layer(x0, x1, sparse_feature_num, dense_feature_num, embedding_size)
    return x1


def fae_dcn_criteosearch(dense_input, hot_sparse_input, cold_sparse_input, cold_category_input, y_, embedding_size=128, enable_push_index=False):
    feature_dimension = 16205514 # 14859910
    learning_rate = 0.00001

    dense_feature_num = 3
    sparse_feature_num = 17

    Embedding = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.0001, name="snd_order_embedding", ctx=ht.cpu(0))
    
    # cold embedding
    cold_sparse_input = ht.embedding_lookup_op(
        Embedding, cold_sparse_input, enable_push_index=enable_push_index, ctx=ht.cpu(0)) # (batch_size, 17, embedding_size)
    cold_sparse_input = ht.array_reshape_op(cold_sparse_input, (-1, embedding_size)) # (batch_size * 17, embedding_size)
    # cold_sparse_input = ht.transpose_op(cold_sparse_input) # (embedding_size, batch_size * 17)
    cold_category_input = ht.array_reshape_op(cold_category_input, (-1, 1)) # (batch_size * 17, 1)
    cold_category_input = ht.broadcastto_op(cold_category_input, cold_sparse_input) # (batch_size * 17, embedding_size)
    # cold_category_input = ht.transpose_op(cold_category_input) # (1, batch_size * 17)

    cold_sparse_input = ht.mul_op(cold_sparse_input, cold_category_input) # (batch_size * 17, embedding_size)

    cold_sparse_input = ht.array_reshape_op(cold_sparse_input, (-1, sparse_feature_num, embedding_size)) # (batch_size, 17, embedding_size)
    cold_sparse_input = ht.reduce_sum_op(cold_sparse_input, axes=1) # (batch_size, embedding_size)

    # hot embedding
    W5 = init.random_normal( # embedding
        [num_hot_emb, embedding_size], stddev=0.001, name="W5")

    hot_sparse_input = ht.one_hot_op(hot_sparse_input, num_hot_emb + 1) # (batch_size, 17, num_hot_emb + 1)
    # "+ 1" is used to store cold_embeddings in index 0
    hot_sparse_input = ht.slice_op(hot_sparse_input, (0, 0, 1), (-1, sparse_feature_num, num_hot_emb)) # (batch_size, 17, num_hot_emb)
    hot_sparse_input = ht.reduce_sum_op(hot_sparse_input, axes=1) # (batch_size, num_hot_emb)
    hot_sparse_input = ht.matmul_op(hot_sparse_input, W5) # (batch_size, embedding_size)

    sparse_input = hot_sparse_input + cold_sparse_input
    # For each sample, the embeddings of all sparse inputs are summed together.
    
    x = ht.concat_op(sparse_input, dense_input, axis=1)
    # Cross Network
    cross_output = build_cross_layer(x, 1, dense_feature_num, embedding_size, num_layers=3) # different from origin dcn

    # DNN
    flatten = x
    W1 = init.random_normal(
        [embedding_size + dense_feature_num, 64], stddev=0.001, name="W1") # different from origin dcn
    W2 = init.random_normal([64, 32], stddev=0.001, name="W2")
    W3 = init.random_normal([32, 16], stddev=0.001, name="W3")

    W4 = init.random_normal(
        [16 + embedding_size + dense_feature_num, 1], stddev=0.001, name="W4") # different from origin dcn

    fc1 = ht.matmul_op(flatten, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    y3 = ht.matmul_op(relu2, W3)

    y4 = ht.concat_op(cross_output, y3, axis=1)
    y = ht.matmul_op(y4, W4)
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
