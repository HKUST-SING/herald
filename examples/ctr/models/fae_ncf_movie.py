import hetu as ht
from hetu import init

def fae_ncf_movie(hot_sparse_user_input, cold_sparse_user_input, hot_sparse_item_input, cold_sparse_item_input,
                  cold_user_category_input, cold_item_category_input, y_, embedding_size=128, enable_push_index=False):
    embed_dim = embedding_size
    layers = [64, 32, 16, 8]
    learning_rate = 0.01
    num_users = 162541
    num_items = 59047
    hot_user_number = 1107
    hot_item_number = 1109

    Embedding_users = init.random_normal(
        (num_users, embed_dim + layers[0] // 2), stddev=0.01, name="embed_user", ctx=ht.cpu(0))

    Embedding_items = init.random_normal(
        (num_items, embed_dim + layers[0] // 2), stddev=0.01, name="embed_item", ctx=ht.cpu(0))

    # cold user
    cold_user_latent = ht.embedding_lookup_op(
        Embedding_users, cold_sparse_user_input, enable_push_index=enable_push_index, ctx=ht.cpu(0))  # (batch_size, 1, embed_dim)
    cold_user_latent = ht.array_reshape_op(cold_user_latent, (-1, embed_dim + layers[0] // 2)) # (batch_size * 1, embedding_size)
    cold_user_category_input = ht.array_reshape_op(cold_user_category_input, (-1, 1)) # (batch_size * 1, 1)
    cold_user_category_input = ht.broadcastto_op(cold_user_category_input, cold_user_latent) # (batch_size * 1, embedding_size)
    cold_user_latent = ht.mul_op(cold_user_latent, cold_user_category_input) # (batch_size * 1, embedding_size)
    
    # hot user
    W5 = init.random_normal((hot_user_number, embed_dim + layers[0] // 2), stddev=0.1, name='W5')
    hot_user_latent = ht.one_hot_op(hot_sparse_user_input, hot_user_number + 1) # (batch_size * 1, hot_user_number + 1)
    # "+ 1" is used to store cold_embeddings in index 0
    hot_user_latent = ht.slice_op(hot_user_latent, (0, 1), (-1, hot_user_number)) # (batch_size * 1, hot_user_number)
    # hot_user_latent = ht.array_reshape_op(hot_user_latent, (-1, hot_user_number)) # (batch_size * 1, hot_user_number)
    hot_user_latent = ht.matmul_op(hot_user_latent, W5) # (batch_size * 1, embedding_size)

    user_latent = cold_user_latent + hot_user_latent # (batch_size * 1, embedding_size)
    
    # cold item
    cold_item_latent = ht.embedding_lookup_op(
        Embedding_items, cold_sparse_item_input, enable_push_index=enable_push_index, ctx=ht.cpu(0))  # (batch_size, 1, embed_dim)
    cold_item_latent = ht.array_reshape_op(cold_item_latent, (-1, embed_dim + layers[0] // 2)) # (batch_size * 1, embedding_size)
    cold_item_category_input = ht.array_reshape_op(cold_item_category_input, (-1, 1)) # (batch_size * 1, 1)
    cold_item_category_input = ht.broadcastto_op(cold_item_category_input, cold_item_latent) # (batch_size * 1, embedding_size)
    cold_item_latent = ht.mul_op(cold_item_latent, cold_item_category_input) # (batch_size * 1, embedding_size)
    
    # hot item
    W6 = init.random_normal((hot_item_number, embed_dim + layers[0] // 2), stddev=0.1, name='W6')
    hot_item_latent = ht.one_hot_op(hot_sparse_item_input, hot_item_number + 1) # (batch_size * 1, hot_item_number + 1)
    # "+ 1" is used to store cold_embeddings in index 0
    hot_item_latent = ht.slice_op(hot_item_latent, (0, 1), (-1, hot_item_number)) # (batch_size * 1, hot_item_number)
    # hot_item_latent = ht.array_reshape_op(hot_item_latent, (-1, hot_item_number)) # (batch_size * hot_item_number)
    hot_item_latent = ht.matmul_op(hot_item_latent, W6) # (batch_size * 1, embedding_size)

    item_latent = cold_item_latent + hot_item_latent # (batch_size * 1, embedding_size)

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

