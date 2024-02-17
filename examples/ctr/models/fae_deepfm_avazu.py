import hetu as ht
from hetu import init

import numpy as np
import time

num_hot_emb = 94492 # hot rate is 1%

def fae_dfm_avazu(dense_input, hot_sparse_input, cold_sparse_input, cold_category_input, y_, embedding_size=128, enable_push_index=False):
    feature_dimension = 9449189 # 33762577
    learning_rate = 0.01

    # FM
    Embedding1 = init.random_normal(
        [feature_dimension, 1], stddev=0.01, name="fst_order_embedding", ctx=ht.cpu(0))
    
    # cold 1d embedding
    cold_sparse_1dim_input = ht.embedding_lookup_op(
        Embedding1, cold_sparse_input, enable_push_index=enable_push_index, ctx=ht.cpu(0)) # (batch_size, 18, 1)
    cold_sparse_1dim_input = ht.array_reshape_op(cold_sparse_1dim_input, (-1, 1)) # (batch_size * 18, 1)
    cold_category_input = ht.array_reshape_op(cold_category_input, (-1, 1)) # (batch_size * 18, 1)
    cold_sparse_1dim_input = ht.mul_op(cold_sparse_1dim_input, cold_category_input) # (batch_size * 18, 1)
    cold_sparse_1dim_input = ht.array_reshape_op(cold_sparse_1dim_input, (-1, 18, 1)) # (batch_size, 18, 1)
    cold_sparse_1dim_input = ht.reduce_sum_op(cold_sparse_1dim_input, axes=1) # (batch_size, 1)
    
    # hot 1d embedding
    W4 = init.random_normal( # embedding
        [num_hot_emb, 1], stddev=0.01, name="W4")

    hot_sparse_input = ht.one_hot_op(hot_sparse_input, num_hot_emb + 1) # (batch_size, 18, num_hot_emb + 1)
    # "+ 1" is used to store cold_embeddings in index 0
    hot_sparse_input = ht.slice_op(hot_sparse_input, (0, 0, 1), (-1, 18, num_hot_emb)) # (batch_size, 18, num_hot_emb)
    hot_sparse_input = ht.reduce_sum_op(hot_sparse_input, axes=1) # (batch_size, num_hot_emb)
    hot_sparse_1dim_input = ht.matmul_op(hot_sparse_input, W4) # (batch_size, 1)

    fm_sparse_part = hot_sparse_1dim_input + cold_sparse_1dim_input # (batch_size, 1)

    FM_W = init.random_normal([4, 1], stddev=0.01, name="dense_parameter")
    fm_dense_part = ht.matmul_op(dense_input, FM_W)
    # fst order output
    y1 = fm_dense_part + fm_sparse_part

    Embedding2 = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding", ctx=ht.cpu(0))
    
    # cold 2d embedding sum
    cold_sparse_2dim_input = ht.embedding_lookup_op(
        Embedding2, cold_sparse_input, enable_push_index=enable_push_index, ctx=ht.cpu(0)) # (batch_size, 18, embedding_size)
    cold_sparse_2dim_input = ht.array_reshape_op(cold_sparse_2dim_input, (-1, embedding_size)) # (batch_size * 18, embedding_size)
    cold_category_input_bc = ht.broadcastto_op(cold_category_input, cold_sparse_2dim_input) # [batch_size * 18, embedding_size]
    cold_sparse_2dim_input = ht.mul_op(cold_sparse_2dim_input, cold_category_input_bc) # (batch_size * 18, embedding_size)
    cold_sparse_2dim_input = ht.array_reshape_op(cold_sparse_2dim_input, (-1, 18, embedding_size)) # (batch_size, 18, embedding_size)
    cold_sparse_2dim_sum = ht.reduce_sum_op(cold_sparse_2dim_input, axes=1) # (batch_size, embedding_size)
    
    # hot 2d embedding sum
    W5 = init.random_normal( # embedding
        [num_hot_emb, embedding_size], stddev=0.01, name="W5")

    hot_sparse_2dim_sum = ht.matmul_op(hot_sparse_input, W5) # (batch_size, embedding_size)

    # sum square
    sparse_2dim_sum = cold_sparse_2dim_sum + hot_sparse_2dim_sum # (batch_size, embedding_size)
    sparse_2dim_sum_square = ht.mul_op(sparse_2dim_sum, sparse_2dim_sum)
    
    # cold 2d embedding square sum
    cold_sparse_2dim_square = ht.mul_op(cold_sparse_2dim_input, cold_sparse_2dim_input) # (batch_size, 18, embedding_size)
    cold_square_2dim_square_sum = ht.reduce_sum_op(cold_sparse_2dim_square, axes=1) # (batch_size, embedding_size)

    # hot 2d embedding square sum
    W5_square = ht.mul_op(W5, W5)
    hot_sparse_2dim_square_sum = ht.matmul_op(hot_sparse_input, W5_square) # (batch_size, embedding_size)

    # square sum
    sparse_2dim_square_sum = cold_square_2dim_square_sum + hot_sparse_2dim_square_sum


    sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
    sparse_2dim_half = sparse_2dim * 0.5
    # snd order output
    y2 = ht.reduce_sum_op(sparse_2dim_half, axes=1, keepdims=True)

    # DNN
    W1 = init.random_normal([embedding_size, 256], stddev=0.01, name="W1") # different from original dfm
    W2 = init.random_normal([256, 256], stddev=0.01, name="W2")
    W3 = init.random_normal([256, 1], stddev=0.01, name="W3")

    fc1 = ht.matmul_op(sparse_2dim_sum, W1)
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
