import tensorflow as tf


def ncf_movie(sparse_input, y_, partitioner=None, part_all=True, param_on_gpu=True):

    embed_dim = 8
    layers = [64, 32, 16, 8]
    num_users = 162541
    num_items = 59047
    learning_rate = 0.01 / 8  # here to comply with HETU

    # feature_dimension = 33762577
    # embedding_size = 128
    all_partitioner, embed_partitioner = (
        partitioner, None) if part_all else (None, partitioner)
    with tf.compat.v1.variable_scope('wdl', dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01), partitioner=all_partitioner):
        with tf.device('/cpu:0'):
            Embedding = tf.compat.v1.get_variable(name="Embedding", shape=(num_users + num_items, embed_dim + layers[0] // 2), partitioner=embed_partitioner)
            sparse_latent = tf.nn.embedding_lookup(Embedding, sparse_input)
        device = '/gpu:0' if param_on_gpu else '/cpu:0'
        with tf.device(device):
            W1 = tf.compat.v1.get_variable(name='W1', shape=[layers[0], layers[1]])
            W2 = tf.compat.v1.get_variable(name='W2', shape=[layers[1], layers[2]])
            W3 = tf.compat.v1.get_variable(name='W3', shape=[layers[2], layers[3]])
            W4 = tf.compat.v1.get_variable(name='W4', shape=[embed_dim + layers[3], 1])
        with tf.device('/gpu:0'):
            user_latent = tf.slice(sparse_latent, [0, 0, 0], [-1, 1, embed_dim + layers[0] // 2])
            item_latent = tf.slice(sparse_latent, [0, 1, 0], [-1, 1, embed_dim + layers[0] // 2])

            user_latent = tf.reshape(user_latent, (-1, embed_dim + layers[0] // 2))
            item_latent = tf.reshape(item_latent, (-1, embed_dim + layers[0] // 2))

            mf_user_latent = tf.slice(user_latent, [0, 0], [-1, embed_dim])
            mlp_user_latent = tf.slice(user_latent, [0, embed_dim], [-1, -1])
            mf_item_latent = tf.slice(item_latent, [0, 0], [-1, embed_dim])
            mlp_item_latent = tf.slice(item_latent, [0, embed_dim], [-1, -1])

            mf_vector = tf.multiply(mf_user_latent, mf_item_latent)
            mlp_vector = tf.concat((mlp_user_latent, mlp_item_latent), 1)
            fc1 = tf.matmul(mlp_vector, W1)
            relu1 = tf.nn.relu(fc1)
            fc2 = tf.matmul(relu1, W2)
            relu2 = tf.nn.relu(fc2)
            fc3 = tf.matmul(relu2, W3)
            relu3 = tf.nn.relu(fc3)
            concat_vector = tf.concat((mf_vector, relu3), 1)
            y = tf.matmul(concat_vector, W4)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate)
            return loss, y, optimizer
