import hetu as ht
from hetu import init

import numpy as np
import time

num_hot_emb = 337626 # hot rate is 1%

def fae_wdl_criteo(dense_input, hot_sparse_input, cold_sparse_input, cold_category_input, y_, embedding_size=128, enable_push_index=False):
# hot_sparse_input (shape: [batch_size, 26]) represents the index (the ranking if feature is hot; 0 if feature is cold) of sparse features of current sample
# cold_sparse_input (shape: [batch_size, 26]) replaces the hot features in original sparse_input with 0
# cold_category_input is a multi-hot encoding of the samples' features (shape: [batch_size, 26]) where hot inputs are encoded into 0 and cold ones are encoded into 1)

    feature_dimension = 33762577
    learning_rate = 0.01
    Embedding = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding", ctx=ht.cpu(0))

    # cold embedding
    cold_sparse_input = ht.embedding_lookup_op(
        Embedding, cold_sparse_input, enable_push_index=enable_push_index, ctx=ht.cpu(0)) # (batch_size, 26, embedding_size)
    cold_sparse_input = ht.array_reshape_op(cold_sparse_input, (-1, embedding_size)) # (batch_size * 26, embedding_size)
    # cold_sparse_input = ht.transpose_op(cold_sparse_input) # (embedding_size, batch_size * 26)
    cold_category_input = ht.array_reshape_op(cold_category_input, (-1, 1)) # (batch_size * 26, 1)
    cold_category_input = ht.broadcastto_op(cold_category_input, cold_sparse_input) # (batch_size * 26, embedding_size)
    # cold_category_input = ht.transpose_op(cold_category_input) # (1, batch_size * 26)

    cold_sparse_input = ht.mul_op(cold_sparse_input, cold_category_input) # (batch_size * 26, embedding_size)

    cold_sparse_input = ht.array_reshape_op(cold_sparse_input, (-1, 26, embedding_size)) # (batch_size, 26, embedding_size)
    cold_sparse_input = ht.reduce_sum_op(cold_sparse_input, axes=1) # (batch_size, embedding_size)

    # hot embedding
    W5 = init.random_normal( # embedding
        [num_hot_emb, embedding_size], stddev=0.01, name="W5")

    hot_sparse_input = ht.one_hot_op(hot_sparse_input, num_hot_emb + 1) # (batch_size, 26, num_hot_emb + 1)
    # "+ 1" is used to store cold_embeddings in index 0
    hot_sparse_input = ht.slice_op(hot_sparse_input, (0, 0, 1), (-1, 26, num_hot_emb)) # (batch_size, 26, num_hot_emb)
    hot_sparse_input = ht.reduce_sum_op(hot_sparse_input, axes=1) # (batch_size, num_hot_emb)
    hot_sparse_input = ht.matmul_op(hot_sparse_input, W5) # (batch_size, embedding_size)

    sparse_input = hot_sparse_input + cold_sparse_input
    # For each sample, the embeddings of all sparse inputs are summed together.
    
    # DNN
    flatten = dense_input
    W1 = init.random_normal([13, 256], stddev=0.01, name="W1")
    W2 = init.random_normal([256, 256], stddev=0.01, name="W2")
    W3 = init.random_normal([256, 256], stddev=0.01, name="W3")

    W4 = init.random_normal(
        [256 + embedding_size, 1], stddev=0.01, name="W4")

    fc1 = ht.matmul_op(flatten, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    y3 = ht.matmul_op(relu2, W3)

    y4 = ht.concat_op(sparse_input, y3, axis=1)
    y = ht.matmul_op(y4, W4)
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
