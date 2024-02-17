import torch, os, pickle, time
import torch.nn as nn
import numpy as np

def process_all_criteo_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/criteo'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]

    # files = [np.load(filename) for filename in file_paths]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]

def criteo_preprocess(hot_rate):
    path = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_criteo')

    start_time = time.time()
    _, sparse, _ = process_all_criteo_data()
    end_time = time.time()
    print("Reading Criteo dataset takes ", end_time - start_time , "s")

    # sparse = (sparse[0][:100], sparse[1][:100])

    num_fea = 26
    num_emb = 33762577
    num_hot_emb = int(hot_rate * num_emb) + 1
    print("Number of hot embeddings is ", num_hot_emb)

    start_time = time.time()
    emb_occurence = np.zeros((num_emb,), dtype=int)
    for i in range(len(sparse[0])): # training set
        for j in range(num_fea):
            emb_occurence[sparse[0][i][j]] += 1

    for i in range(len(sparse[1])): # test set
        for j in range(num_fea):
            emb_occurence[sparse[1][i][j]] += 1
    end_time = time.time()
    print("Counting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "emb_occurence.npy")
    np.save(target_path, emb_occurence)
    print("Embedding occurence has been saved in ", target_path)

    start_time = time.time()
    top_n_indexes = np.argpartition(emb_occurence, -num_hot_emb)[-num_hot_emb:]
    hot_dict = {}
    for ranking, embedding_id in enumerate(top_n_indexes):
        hot_dict[embedding_id] = ranking + 1 # 0 is reserved for cold features, which will be discarded in training
    end_time = time.time()
    print("Sorting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "hot_dict.json")
    with open(target_path, 'wb') as fp:
        pickle.dump(hot_dict, fp)
    print("Hot embedding dictionary has been saved in ", target_path)

    start_time = time.time()
    train_hot_sparse_input = np.zeros(sparse[0].shape, dtype=int)
    train_cold_sparse_input = np.zeros(sparse[0].shape, dtype=int)
    train_cold_category = np.zeros(sparse[0].shape, dtype=int)
    for i in range(len(sparse[0])):
        for j in range(num_fea):
            if (sparse[0][i][j] in hot_dict): # hot
                train_hot_sparse_input[i][j] = hot_dict[sparse[0][i][j]]
                train_cold_sparse_input[i][j] = 0
                train_cold_category[i][j] = 0
            
            else: # cold
                train_hot_sparse_input[i][j] = 0
                train_cold_sparse_input[i][j] = sparse[0][i][j]
                train_cold_category[i][j] = 1
    end_time = time.time()
    print("Training data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "train_hot_sparse_input")
    np.save(target_path, train_hot_sparse_input)
    print("Training hot data has been saved in ", target_path)

    target_path = os.path.join(path, "train_cold_sparse_input")
    np.save(target_path, train_cold_sparse_input)
    print("Training cold data has been saved in ", target_path)
    
    target_path = os.path.join(path, "train_cold_category")
    np.save(target_path, train_cold_category)
    print("Training cold category has been saved in ", target_path)

    start_time = time.time()
    test_hot_sparse_input = np.zeros(sparse[1].shape, dtype=int)
    test_cold_sparse_input = np.zeros(sparse[1].shape, dtype=int)
    test_cold_category = np.zeros(sparse[1].shape, dtype=int)

    for i in range(len(sparse[1])):
        for j in range(num_fea):
            if (sparse[1][i][j] in hot_dict): # hot
                test_hot_sparse_input[i][j] = hot_dict[sparse[1][i][j]]
                test_cold_sparse_input[i][j] = 0
                test_cold_category[i][j] = 0
            
            else: # cold
                test_hot_sparse_input[i][j] = 0
                test_cold_sparse_input[i][j] = sparse[1][i][j]
                test_cold_category[i][j] = 1
    end_time = time.time()
    print("Testing data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "test_hot_sparse_input")
    np.save(target_path, test_hot_sparse_input)
    print("Testing hot data has been saved in ", target_path)

    target_path = os.path.join(path, "test_cold_sparse_input")
    np.save(target_path, test_cold_sparse_input)
    print("Testing cold data has been saved in ", target_path)

    target_path = os.path.join(path, "test_cold_category")
    np.save(target_path, test_cold_category)
    print("Testing cold category has been saved in ", target_path)

    print("FAE criteo data preprocessing done.")

def process_all_avazu_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/avazu'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]

    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]

def avazu_preprocess(hot_rate):
    path = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_avazu')

    start_time = time.time()
    _, sparse, _ = process_all_avazu_data()
    end_time = time.time()
    print("Reading Avazu dataset takes ", end_time - start_time , "s")

    num_fea = 18
    num_emb = 9449189
    num_hot_emb = int(hot_rate * num_emb) + 1
    print("Number of hot embeddings is ", num_hot_emb)

    start_time = time.time()
    emb_occurence = np.zeros((num_emb,), dtype=int)
    for i in range(len(sparse[0])): # training set
        for j in range(num_fea):
            emb_occurence[sparse[0][i][j]] += 1

    for i in range(len(sparse[1])): # test set
        for j in range(num_fea):
            emb_occurence[sparse[1][i][j]] += 1
    end_time = time.time()
    print("Counting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "emb_occurence.npy")
    np.save(target_path, emb_occurence)
    print("Embedding occurence has been saved in ", target_path)

    start_time = time.time()
    top_n_indexes = np.argpartition(emb_occurence, -num_hot_emb)[-num_hot_emb:]
    hot_dict = {}
    for ranking, embedding_id in enumerate(top_n_indexes):
        hot_dict[embedding_id] = ranking + 1 # 0 is reserved for cold features, which will be discarded in training
    end_time = time.time()
    print("Sorting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "hot_dict.json")
    with open(target_path, 'wb') as fp:
        pickle.dump(hot_dict, fp)
    print("Hot embedding dictionary has been saved in ", target_path)

    start_time = time.time()
    train_hot_sparse_input = np.zeros(sparse[0].shape, dtype=int)
    train_cold_sparse_input = np.zeros(sparse[0].shape, dtype=int)
    train_cold_category = np.zeros(sparse[0].shape, dtype=int)
    for i in range(len(sparse[0])):
        for j in range(num_fea):
            if (sparse[0][i][j] in hot_dict): # hot
                train_hot_sparse_input[i][j] = hot_dict[sparse[0][i][j]]
                train_cold_sparse_input[i][j] = 0
                train_cold_category[i][j] = 0
            
            else: # cold
                train_hot_sparse_input[i][j] = 0
                train_cold_sparse_input[i][j] = sparse[0][i][j]
                train_cold_category[i][j] = 1
    end_time = time.time()
    print("Training data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "train_hot_sparse_input")
    np.save(target_path, train_hot_sparse_input)
    print("Training hot data has been saved in ", target_path)

    target_path = os.path.join(path, "train_cold_sparse_input")
    np.save(target_path, train_cold_sparse_input)
    print("Training cold data has been saved in ", target_path)
    
    target_path = os.path.join(path, "train_cold_category")
    np.save(target_path, train_cold_category)
    print("Training cold category has been saved in ", target_path)

    start_time = time.time()
    test_hot_sparse_input = np.zeros(sparse[1].shape, dtype=int)
    test_cold_sparse_input = np.zeros(sparse[1].shape, dtype=int)
    test_cold_category = np.zeros(sparse[1].shape, dtype=int)

    for i in range(len(sparse[1])):
        for j in range(num_fea):
            if (sparse[1][i][j] in hot_dict): # hot
                test_hot_sparse_input[i][j] = hot_dict[sparse[1][i][j]]
                test_cold_sparse_input[i][j] = 0
                test_cold_category[i][j] = 0
            
            else: # cold
                test_hot_sparse_input[i][j] = 0
                test_cold_sparse_input[i][j] = sparse[1][i][j]
                test_cold_category[i][j] = 1
    end_time = time.time()
    print("Testing data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "test_hot_sparse_input")
    np.save(target_path, test_hot_sparse_input)
    print("Testing hot data has been saved in ", target_path)

    target_path = os.path.join(path, "test_cold_sparse_input")
    np.save(target_path, test_cold_sparse_input)
    print("Testing cold data has been saved in ", target_path)

    target_path = os.path.join(path, "test_cold_category")
    np.save(target_path, test_cold_category)
    print("Testing cold category has been saved in ", target_path)

    print("FAE avazu data preprocessing done.")

def process_all_criteo_search_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/criteo_search'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]
    if not all([os.path.exists(p) for p in file_paths]):
        raise PermissionError("No criteo search dataset available")
        # download_avazu(path)
    # files = [np.load(filename) for filename in file_paths]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]

def criteo_search_preprocess(hot_rate):
    path = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_criteo_search')

    start_time = time.time()
    _, sparse, _ = process_all_criteo_search_data()
    end_time = time.time()
    print("Reading criteo_search dataset takes ", end_time - start_time , "s")

    num_fea = 17
    num_emb = 16205514
    num_hot_emb = int(hot_rate * num_emb) + 1
    print("Number of hot embeddings is ", num_hot_emb)

    start_time = time.time()
    emb_occurence = np.zeros((num_emb,), dtype=int)
    for i in range(len(sparse[0])): # training set
        for j in range(num_fea):
            emb_occurence[sparse[0][i][j]] += 1

    for i in range(len(sparse[1])): # test set
        for j in range(num_fea):
            emb_occurence[sparse[1][i][j]] += 1
    end_time = time.time()
    print("Counting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "emb_occurence.npy")
    np.save(target_path, emb_occurence)
    print("Embedding occurence has been saved in ", target_path)

    start_time = time.time()
    top_n_indexes = np.argpartition(emb_occurence, -num_hot_emb)[-num_hot_emb:]
    hot_dict = {}
    for ranking, embedding_id in enumerate(top_n_indexes):
        hot_dict[embedding_id] = ranking + 1 # 0 is reserved for cold features, which will be discarded in training
    end_time = time.time()
    print("Sorting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "hot_dict.json")
    with open(target_path, 'wb') as fp:
        pickle.dump(hot_dict, fp)
    print("Hot embedding dictionary has been saved in ", target_path)

    start_time = time.time()
    train_hot_sparse_input = np.zeros(sparse[0].shape, dtype=int)
    train_cold_sparse_input = np.zeros(sparse[0].shape, dtype=int)
    train_cold_category = np.zeros(sparse[0].shape, dtype=int)
    for i in range(len(sparse[0])):
        for j in range(num_fea):
            if (sparse[0][i][j] in hot_dict): # hot
                train_hot_sparse_input[i][j] = hot_dict[sparse[0][i][j]]
                train_cold_sparse_input[i][j] = 0
                train_cold_category[i][j] = 0
            
            else: # cold
                train_hot_sparse_input[i][j] = 0
                train_cold_sparse_input[i][j] = sparse[0][i][j]
                train_cold_category[i][j] = 1
    end_time = time.time()
    print("Training data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "train_hot_sparse_input")
    np.save(target_path, train_hot_sparse_input)
    print("Training hot data has been saved in ", target_path)

    target_path = os.path.join(path, "train_cold_sparse_input")
    np.save(target_path, train_cold_sparse_input)
    print("Training cold data has been saved in ", target_path)
    
    target_path = os.path.join(path, "train_cold_category")
    np.save(target_path, train_cold_category)
    print("Training cold category has been saved in ", target_path)

    start_time = time.time()
    test_hot_sparse_input = np.zeros(sparse[1].shape, dtype=int)
    test_cold_sparse_input = np.zeros(sparse[1].shape, dtype=int)
    test_cold_category = np.zeros(sparse[1].shape, dtype=int)

    for i in range(len(sparse[1])):
        for j in range(num_fea):
            if (sparse[1][i][j] in hot_dict): # hot
                test_hot_sparse_input[i][j] = hot_dict[sparse[1][i][j]]
                test_cold_sparse_input[i][j] = 0
                test_cold_category[i][j] = 0
            
            else: # cold
                test_hot_sparse_input[i][j] = 0
                test_cold_sparse_input[i][j] = sparse[1][i][j]
                test_cold_category[i][j] = 1
    end_time = time.time()
    print("Testing data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "test_hot_sparse_input")
    np.save(target_path, test_hot_sparse_input)
    print("Testing hot data has been saved in ", target_path)

    target_path = os.path.join(path, "test_cold_sparse_input")
    np.save(target_path, test_cold_sparse_input)
    print("Testing cold data has been saved in ", target_path)

    target_path = os.path.join(path, "test_cold_category")
    np.save(target_path, test_cold_category)
    print("Testing cold category has been saved in ", target_path)

    print("FAE criteo_search data preprocessing done.")

def process_all_movie_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/ml-25m')): # ../../rec/datasets/ml-25m
    file_paths = [os.path.join(path, data)
                  for data in ['train.npz', 'test.npy']]
    return np.load(file_paths[0]), np.load(file_paths[1])

def movie_preprocess(hot_rate):
    path = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_ml-25m')

    start_time = time.time()
    train, _ = process_all_movie_data()
    end_time = time.time()
    print("Reading movie dataset takes ", end_time - start_time , "s")

    train_sparse_user = train['user_input'] # range in [0, 162540], 162541
    train_sparse_item = train['item_input'] # range in [162541, 221587], 59047

    num_emb = 221588
    num_hot_emb = int(hot_rate * num_emb) + 1
    print("Number of hot embeddings is ", num_hot_emb)

    start_time = time.time()
    emb_occurence = np.zeros((num_emb,), dtype=int)
    for i in range(len(train_sparse_user)): # user set
        emb_occurence[train_sparse_user[i]] += 1

    for i in range(len(train_sparse_item)): # item set
        emb_occurence[train_sparse_item[i]] += 1
    end_time = time.time()
    print("Counting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "emb_occurence.npy")
    np.save(target_path, emb_occurence)
    print("Embedding occurence has been saved in ", target_path)

    start_time = time.time()
    top_n_indexes = np.argpartition(emb_occurence, -num_hot_emb)[-num_hot_emb:]

    hot_user_number = 0
    hot_item_number = 0
    hot_user_dict = {}
    hot_item_dict = {}

    for ranking, embedding_id in enumerate(top_n_indexes):
        if embedding_id > 162540: # item
            hot_item_number += 1
            hot_item_dict[embedding_id] = ranking - hot_user_number + 1 # 0 is reserved for cold features.
            # Moreover, we need to subtract the number of existing hot users to get the ranking within items.
        else: # user
            hot_user_number += 1
            hot_user_dict[embedding_id] = ranking - hot_item_number + 1 # Save as mentioned above

    print("hot_user_number is ", hot_user_number)
    print("hot_item_number is ", hot_item_number)
    end_time = time.time()
    print("Sorting embedding occurences takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "hot_user_dict.json")
    with open(target_path, 'wb') as fp:
        pickle.dump(hot_user_dict, fp)
    print("Hot user embedding dictionary has been saved in ", target_path)

    target_path = os.path.join(path, "hot_item_dict.json")
    with open(target_path, 'wb') as fp:
        pickle.dump(hot_item_dict, fp)
    print("Hot item embedding dictionary has been saved in ", target_path)

    start_time = time.time()
    train_hot_sparse_user_input = np.zeros(train_sparse_user.shape, dtype=int)
    train_cold_sparse_user_input = np.zeros(train_sparse_user.shape, dtype=int)
    train_cold_user_category = np.zeros(train_sparse_user.shape, dtype=int)

    train_hot_sparse_item_input = np.zeros(train_sparse_item.shape, dtype=int)
    train_cold_sparse_item_input = np.zeros(train_sparse_item.shape, dtype=int)
    train_cold_item_category = np.zeros(train_sparse_item.shape, dtype=int)

    for i in range(len(train_sparse_user)):
        # user
        if (train_sparse_user[i] in hot_user_dict): # hot
            train_hot_sparse_user_input[i] = hot_user_dict[train_sparse_user[i]]
            train_cold_sparse_user_input[i] = 0
            train_cold_user_category[i] = 0
        
        else: # cold
            train_hot_sparse_user_input[i] = 0
            train_cold_sparse_user_input[i] = train_sparse_user[i]
            train_cold_user_category[i] = 1
        
        # item
        if (train_sparse_item[i] in hot_item_dict): # hot
            train_hot_sparse_item_input[i] = hot_item_dict[train_sparse_item[i]]
            train_cold_sparse_item_input[i] = 0
            train_cold_item_category[i] = 0
        
        else: # cold
            train_hot_sparse_item_input[i] = 0
            train_cold_sparse_item_input[i] = train_sparse_item[i]
            train_cold_item_category[i] = 1

    end_time = time.time()
    print("Training data construction takes ", end_time - start_time , "s")

    target_path = os.path.join(path, "train_hot_sparse_user_input")
    np.save(target_path, train_hot_sparse_user_input)
    print("Training hot data for user has been saved in ", target_path)

    target_path = os.path.join(path, "train_cold_sparse_user_input")
    np.save(target_path, train_cold_sparse_user_input)
    print("Training cold data for user has been saved in ", target_path)
    
    target_path = os.path.join(path, "train_cold_user_category")
    np.save(target_path, train_cold_user_category)
    print("Training cold category for user has been saved in ", target_path)

    target_path = os.path.join(path, "train_hot_sparse_item_input")
    np.save(target_path, train_hot_sparse_item_input)
    print("Training hot data for item has been saved in ", target_path)

    target_path = os.path.join(path, "train_cold_sparse_item_input")
    np.save(target_path, train_cold_sparse_item_input)
    print("Training cold data for item has been saved in ", target_path)
    
    target_path = os.path.join(path, "train_cold_item_category")
    np.save(target_path, train_cold_item_category)
    print("Training cold category for item has been saved in ", target_path)

    print("FAE criteo_search data preprocessing done.")


if __name__=="__main__":
    hot_rate = 0.01
    criteo_preprocess(hot_rate)
    avazu_preprocess(hot_rate)
    criteo_search_preprocess(hot_rate)
    movie_preprocess(hot_rate)