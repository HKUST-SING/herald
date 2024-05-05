import hetu as ht
from hetu import init

import numpy as np


def ncf_movie(sparse_input, y_, embedding_size=128, enable_push_index=False):
    embed_dim = embedding_size
    layers = [64, 32, 16, 8]
    learning_rate = 0.01
    num_users = 162541
    num_items = 59047

    Embedding = init.random_normal(
        (num_users + num_items, embed_dim + layers[0] // 2), stddev=0.01, name="embed", ctx=ht.cpu(0))

    sparse_latent = ht.embedding_lookup_op(
        Embedding, sparse_input, ctx=ht.cpu(0), enable_push_index=enable_push_index)
    user_latent = ht.slice_op(sparse_latent, (0, 0, 0), (-1, 1, embed_dim + layers[0] // 2))
    item_latent = ht.slice_op(sparse_latent, (0, 1, 0), (-1, 1, embed_dim + layers[0] // 2))

    user_latent = ht.array_reshape_op(user_latent, (-1, embed_dim + layers[0] // 2))
    item_latent = ht.array_reshape_op(item_latent, (-1, embed_dim + layers[0] // 2))

    mf_user_latent = ht.slice_op(user_latent, (0, 0), (-1, embed_dim))
    mlp_user_latent = ht.slice_op(user_latent, (0, embed_dim), (-1, -1))
    mf_item_latent = ht.slice_op(item_latent, (0, 0), (-1, embed_dim))
    mlp_item_latent = ht.slice_op(item_latent, (0, embed_dim), (-1, -1))

    W1 = init.random_normal((layers[0], layers[1]), stddev=0.1, name='W1')
    W2 = init.random_normal((layers[1], layers[2]), stddev=0.1, name='W2')
    W3 = init.random_normal((layers[2], layers[3]), stddev=0.1, name='W3')
    W4 = init.random_normal((embed_dim + layers[3], 1), stddev=0.1, name='W4')

    mf_vector = ht.mul_op(mf_user_latent, mf_item_latent)
    mlp_vector = ht.concat_op(mlp_user_latent, mlp_item_latent, axis=1)
    fc1 = ht.matmul_op(mlp_vector, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    fc3 = ht.matmul_op(relu2, W3)
    relu3 = ht.relu_op(fc3)
    concat_vector = ht.concat_op(mf_vector, relu3, axis=1)
    y = ht.matmul_op(concat_vector, W4)
    y = ht.sigmoid_op(y)
    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    return loss, y, y_, train_op
