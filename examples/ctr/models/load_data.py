import os
import numpy as np

###########################################################################
# Avazu
###########################################################################
def download_avazu(path):
    import tarfile
    import pandas as pd
    from six.moves import urllib
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.isdir(path), 'Please provide a directory path.'
    # this source may be invalid, please use other valid sources.
    # origin = (
    #     'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'
    # )
    # print('Downloading data from %s' % origin)
    # dataset = os.path.join(path, 'criteo.tar.gz')
    # urllib.request.urlretrieve(origin, dataset)
    # print("Extracting criteo zip...")
    # with tarfile.open(dataset) as f:
    #     f.extractall(path=path)
    print("Create local files...")

    # save csv filed
    df = pd.read_csv(os.path.join(path, "train.csv"), low_memory=False, sep=',')
    df.columns = ['id','click','I1','C1','I2','C2','C3','C4',
                  'C5','C6','C7','C8','C9','C10','I3','I4',
                  'C11','C12','C13','C14','C15','C16','C17','C18']
    # df.to_csv(os.path.join(path, "train.csv"), index=0)
    print('Csv file read.')

    # save numpy arrays
    target_path = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'train_sparse_feats.npy', 'train_labels.npy',
        'test_dense_feats.npy', 'test_sparse_feats.npy', 'test_labels.npy']]
    dense_feats = [col for col in df.columns if col.startswith('I')]
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    labels = df['click']
    dense_feats = process_dense_feats(df, dense_feats)
    sparse_feats = process_sparse_feats(df, sparse_feats)
    num_data = dense_feats.shape[0]
    perm = np.random.permutation(num_data)
    # split data in 2 parts
    test_num = num_data // 10
    processed_data = [
        dense_feats.loc[perm[:-test_num]],  # train dense
        sparse_feats.loc[perm[:-test_num]],  # train sparse
        labels.loc[perm[:-test_num]],       # train labels
        dense_feats.loc[perm[-test_num:]],  # validate dense
        sparse_feats.loc[perm[-test_num:]],  # validate sparse
        labels.loc[perm[-test_num:]],       # validate labels  
    ]
    print('Array shapes:')
    for i in range(len(processed_data)):
        print(os.path.split(target_path[i])
              [-1].split('.')[0], processed_data[i].shape)
        np.save(target_path[i], processed_data[i])
    print('Numpy arrays saved.')

###########################################################################
# Criteo Search
###########################################################################
def download_criteo_search(path):
    import tarfile
    import pandas as pd
    from six.moves import urllib
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.isdir(path), 'Please provide a directory path.'
    # this source may be invalid, please use other valid sources.
    # origin = (
    #     'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'
    # )
    # print('Downloading data from %s' % origin)
    # dataset = os.path.join(path, 'criteo.tar.gz')
    # urllib.request.urlretrieve(origin, dataset)
    # print("Extracting criteo zip...")
    # with tarfile.open(dataset) as f:
    #     f.extractall(path=path)
    print("Create local files...")

    # save csv filed
    df = pd.read_csv(os.path.join(path, "CriteoSearchData"), low_memory=False, sep='\t', header=None)
    df.columns = ['sale', 'salesamount', 'timedelay'] + ["I" + 
                 str(i) for i in range(1, 4)] + ["C" + str(i) for i in range(4, 21)]
    # df.to_csv(os.path.join(path, "train.csv"), index=0)
    print('Csv file read.')

    # save numpy arrays
    target_path = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'train_sparse_feats.npy', 'train_labels.npy',
        'test_dense_feats.npy', 'test_sparse_feats.npy', 'test_labels.npy']]
    
    labels = df['sale']
    dense_feats = [col for col in df.columns if col.startswith('I')]
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    dense_feats = process_dense_feats_search(df, dense_feats)
    sparse_feats = process_sparse_feats(df, sparse_feats)
    num_data = dense_feats.shape[0]
    perm = np.random.permutation(num_data)
    # split data in 2 parts
    test_num = num_data // 10
    processed_data = [
        dense_feats.loc[perm[:-test_num]],  # train dense
        sparse_feats.loc[perm[:-test_num]],  # train sparse
        labels.loc[perm[:-test_num]],       # train labels
        dense_feats.loc[perm[-test_num:]],  # validate dense
        sparse_feats.loc[perm[-test_num:]],  # validate sparse
        labels.loc[perm[-test_num:]],       # validate labels  
    ]
    print('Array shapes:')
    for i in range(len(processed_data)):
        print(os.path.split(target_path[i])
              [-1].split('.')[0], processed_data[i].shape)
        np.save(target_path[i], processed_data[i])
    print('Numpy arrays saved.')

###########################################################################
# criteo
###########################################################################

def download_criteo(path):
    import tarfile
    import pandas as pd
    from six.moves import urllib
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.isdir(path), 'Please provide a directory path.'
    # this source may be invalid, please use other valid sources.
    # origin = (
    #     'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'
    # )
    # print('Downloading data from %s' % origin)
    # dataset = os.path.join(path, 'criteo.tar.gz')
    # urllib.request.urlretrieve(origin, dataset)
    # print("Extracting criteo zip...")
    # with tarfile.open(dataset) as f:
    #     f.extractall(path=path)
    print("Create local files...")

    # save csv filed
    df = pd.read_csv(os.path.join(path, "train.txt"), sep='\t', header=None)
    df.columns = ['label'] + ["I" +
                              str(i) for i in range(1, 14)] + ["C"+str(i) for i in range(14, 40)]
    df.to_csv(os.path.join(path, "train.csv"), index=0)
    print('Csv file saved.')

    # save numpy arrays
    target_path = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'train_sparse_feats.npy', 'train_labels.npy',
        'test_dense_feats.npy', 'test_sparse_feats.npy', 'test_labels.npy']]
    dense_feats = [col for col in df.columns if col.startswith('I')]
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    labels = df['label']
    dense_feats = process_dense_feats(df, dense_feats)
    sparse_feats = process_sparse_feats(df, sparse_feats)
    num_data = dense_feats.shape[0]
    perm = np.random.permutation(num_data)
    # split data in 2 parts
    test_num = num_data // 10
    processed_data = [
        dense_feats.loc[perm[:-test_num]],  # train dense
        sparse_feats.loc[perm[:-test_num]],  # train sparse
        labels.loc[perm[:-test_num]],       # train labels
        dense_feats.loc[perm[-test_num:]],  # validate dense
        sparse_feats.loc[perm[-test_num:]],  # validate sparse
        labels.loc[perm[-test_num:]],       # validate labels  
    ]
    print('Array shapes:')
    for i in range(len(processed_data)):
        print(os.path.split(target_path[i])
              [-1].split('.')[0], processed_data[i].shape)
        np.save(target_path[i], processed_data[i])
    print('Numpy arrays saved.')


def process_dense_feats(data, feats):
    d = data.copy()
    d = d[feats].fillna(0.0)
    for f in feats:
        d[f] = d[f].apply(lambda x: np.log(x+1) if x > -1 else -1)
    return d

def process_dense_feats_search(data, feats):
    d = data.copy()
    d = d[feats].fillna(0.0)
    for f in feats:
        d[f] = d[f].apply(lambda x: np.log(float(x)+1) if float(x) > -1 else 0.0)
    return d

def process_sparse_feats(data, feats):
    from sklearn.preprocessing import LabelEncoder
    # process to embeddings.
    d = data.copy()
    d = d[feats].fillna("-1")
    for f in feats:
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])
    feature_cnt = 0
    for f in feats:
        d[f] += feature_cnt
        feature_cnt += d[f].nunique()
    return d


def process_head_criteo_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/criteo'), nrows=20000, return_val=True):
    import pandas as pd
    csv_path = os.path.join(path, "train.csv")
    if not os.path.exists(csv_path):
        download_criteo(path)
    df = pd.read_csv(csv_path, nrows=nrows, header=0)
    dense_feats = [col for col in df.columns if col.startswith('I')]
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    labels = np.array(df['label']).reshape(-1, 1)
    dense_feats = np.array(process_dense_feats(df, dense_feats))
    sparse_feats = np.array(process_sparse_feats(
        df, sparse_feats)).astype(np.int32)
    if return_val:
        test_num = nrows // 10
        train_dense = dense_feats[:-test_num]
        train_sparse = sparse_feats[:-test_num]
        train_label = labels[:-test_num]
        validate_dense = dense_feats[-test_num:]
        validate_sparse = sparse_feats[-test_num:]
        validate_label = labels[-test_num:]
        return (train_dense, validate_dense), (train_sparse, validate_sparse), (train_label, validate_label)
    else:
        return dense_feats, sparse_feats, labels


def process_sampled_criteo_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/criteo')):
    # all data should be available! no checking.
    processed_data = [np.load(os.path.join(path, filename))
                      for filename in ['sampled_dense_feats.npy', 'sampled_sparse_feats.npy', 'sampled_labels.npy']]
    return tuple(processed_data)

def process_all_avazu_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/avazu'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]
    if not all([os.path.exists(p) for p in file_paths]):
        raise PermissionError("No avazu dataset available")
        # download_avazu(path)
    # files = [np.load(filename) for filename in file_paths]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]

def process_all_avazu_data_in_fae_mode(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_avazu'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_hot_sparse_input.npy', 'test_hot_sparse_input.npy',
        'train_cold_sparse_input.npy', 'test_cold_sparse_input.npy', 'train_cold_category.npy', 'test_cold_category.npy',
        'train_labels.npy', 'test_labels.npy']]
    # if not all([os.path.exists(p) for p in file_paths]):
    #     download_criteo(path)
    # files = [np.load(filename) for filename in file_paths]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5]), (files[6], files[7]), (files[8], files[9])
    else:
        return files[0], files[2], files[4], files[6], files[8]

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

def process_all_criteo_search_data_in_fae_mode(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_criteo_search'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_hot_sparse_input.npy', 'test_hot_sparse_input.npy',
        'train_cold_sparse_input.npy', 'test_cold_sparse_input.npy', 'train_cold_category.npy', 'test_cold_category.npy',
        'train_labels.npy', 'test_labels.npy']]
    # if not all([os.path.exists(p) for p in file_paths]):
    #     download_criteo(path)
    # files = [np.load(filename) for filename in file_paths]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5]), (files[6], files[7]), (files[8], files[9])
    else:
        return files[0], files[2], files[4], files[6], files[8]

def process_all_criteo_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/criteo'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]
    if not all([os.path.exists(p) for p in file_paths]):
        download_criteo(path)
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]

def process_all_criteo_data_in_fae_mode(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_criteo'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_hot_sparse_input.npy', 'test_hot_sparse_input.npy',
        'train_cold_sparse_input.npy', 'test_cold_sparse_input.npy', 'train_cold_category.npy', 'test_cold_category.npy',
        'train_labels.npy', 'test_labels.npy']]
    # if not all([os.path.exists(p) for p in file_paths]):
    #     download_criteo(path)
    # files = [np.load(filename) for filename in file_paths]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5]), (files[6], files[7]), (files[8], files[9])
    else:
        return files[0], files[2], files[4], files[6], files[8]

def process_all_movie_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/ml-25m'), return_val=True): # ../../rec/datasets/ml-25m
    file_paths = [os.path.join(path, data)
                  for data in ['train.npz', 'test.npy']]
    train = np.load(file_paths[0])
    test = np.load(file_paths[1])

    trainUsers = train['user_input'].reshape(-1, 1)
    trainItems = train['item_input'].reshape(-1, 1)
    trainSparse = np.concatenate((trainUsers, trainItems), axis=1)
    trainLabels = train['labels'].reshape(-1, 1)

    assert return_val == False, "Not support yet"

    '''
    testUsers = test['user_input'].reshape(-1, 1)
    testItems = test['item_input'].reshape(-1, 1)
    testSparse = np.concatenate((testUsers, testItems), axis=1)
    testLabels = test['labels'].reshape(-1, 1)
    '''

    return trainSparse, trainLabels

def process_all_movie_data_in_fae_mode(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/fae_ml-25m')): # ../../rec/datasets/ml-25m
    file_paths = [os.path.join(path, data)
                  for data in ['train_hot_sparse_user_input.npy', 'train_cold_sparse_user_input.npy',
                               'train_hot_sparse_item_input.npy', 'train_cold_sparse_item_input.npy',
                               'train_cold_user_category.npy',    'train_cold_item_category.npy']]
    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    return np.load(os.path.join(path, 'train.npz')), files[0], files[1], files[2], files[3], files[4], files[5]

###########################################################################
# adult
###########################################################################

def maybe_download(train_data, test_data):
    import pandas as pd
    """if adult data "train.csv" and "test.csv" are not in your directory,
    download them.
    """

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    if not os.path.exists(train_data):
        print("downloading training data...")
        df_train = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                               names=COLUMNS, skipinitialspace=True)
    else:
        df_train = pd.read_csv("train.csv")

    if not os.path.exists(test_data):
        print("downloading testing data...")
        df_test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                              names=COLUMNS, skipinitialspace=True, skiprows=1)
    else:
        df_test = pd.read_csv("test.csv")

    return df_train, df_test


def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


def val2idx(df, cols):
    """helper to index categorical columns before embeddings.
    """
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()

    return df, unique_vals


def onehot(x):
    from sklearn.preprocessing import OneHotEncoder
    return np.array(OneHotEncoder().fit_transform(x).todense())


def wide(df_train, df_test, wide_cols, x_cols, target):
    import pandas as pd
    print('Processing wide data')
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_wide = pd.concat([df_train, df_test])

    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        df_wide.select_dtypes(include=['object']).columns)

    wide_cols += list(crossed_columns_d.keys())

    for k, v in crossed_columns_d.items():
        df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x), axis=1)

    df_wide = df_wide[wide_cols + [target] + ['IS_TRAIN']]

    dummy_cols = [
        c for c in wide_cols if c in categorical_columns + list(crossed_columns_d.keys())]
    df_wide = pd.get_dummies(df_wide, columns=[x for x in dummy_cols])

    train = df_wide[df_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_wide[df_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    assert all(train.columns == test.columns)

    cols = [c for c in train.columns if c != target]
    X_train = train[cols].values
    y_train = train[target].values.reshape(-1, 1)
    X_test = test[cols].values
    y_test = test[target].values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def load_adult_data(return_val=True):
    import pandas as pd
    df_train, df_test = maybe_download("train.csv", "test.csv")

    df_train['income_label'] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test['income_label'] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    age_groups = [0, 25, 65, 90]
    age_labels = range(len(age_groups) - 1)
    df_train['age_group'] = pd.cut(
        df_train['age'], age_groups, labels=age_labels)
    df_test['age_group'] = pd.cut(
        df_test['age'], age_groups, labels=age_labels)

    # columns for wide model
    wide_cols = ['workclass', 'education', 'marital_status', 'occupation',
                 'relationship', 'race', 'gender', 'native_country', 'age_group']
    x_cols = (['education', 'occupation'], ['native_country', 'occupation'])

    # columns for deep model
    embedding_cols = ['workclass', 'education', 'marital_status', 'occupation',
                      'relationship', 'race', 'gender', 'native_country']
    cont_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

    target = 'income_label'

    x_train_wide, y_train_wide, x_test_wide, y_test_wide = wide(
        df_train, df_test, wide_cols, x_cols, target)
    x_train_wide = np.array(x_train_wide).astype(np.float32)
    x_test_wide = np.array(x_test_wide).astype(np.float32)

    print('Processing deep data')
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])

    deep_cols = embedding_cols + cont_cols
    df_deep = df_deep[deep_cols + [target, 'IS_TRAIN']]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_deep[cont_cols] = pd.DataFrame(scaler.fit_transform(df_train[cont_cols]),
                                      columns=cont_cols)
    df_deep, unique_vals = val2idx(df_deep, embedding_cols)

    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

    x_train_deep = np.array([train[c] for c in deep_cols]).astype(np.float32)
    y_train = np.array(train[target].values).reshape(-1, 1).astype(np.int32)
    x_test_deep = np.array([test[c] for c in deep_cols]).astype(np.float32)
    y_test = np.array(test[target].values).reshape(-1, 1).astype(np.int32)

    x_train_deep = np.transpose(x_train_deep)
    x_test_deep = np.transpose(x_test_deep)
    y_train = onehot(y_train)
    y_test = onehot(y_test)

    if return_val:
        return x_train_deep, x_train_wide, y_train, x_test_deep, x_test_wide, y_test
    else:
        return x_train_deep, x_train_wide, y_train


###########################################################################
# avazu
###########################################################################

def process_avazu(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/avazu')):
    import pandas as pd
    # please download in advance from https://www.kaggle.com/c/avazu-ctr-prediction/data
    train_file = os.path.join(path, 'train.csv')
    # test_file = os.path.join(path, 'test.csv') # useless, no labels

    df_train = pd.read_csv(train_file)
    sparse_feats = process_sparse_feats(df_train, df_train.columns[2:])
    # the embedding num for each feature:
    # [240, 7, 7, 4737, 7745, 26, 8552, 559, 36, 2686408, 6729486, 8251, 5, 4, 2626, 8, 9, 435, 4, 68, 172, 60]
    # sum: 9449445

    np.save(os.path.join(path, 'sparse.npy'), sparse_feats)


if __name__ == '__main__':
    download_criteo(os.path.join(os.path.split(
        os.path.abspath(__file__))[0], '../datasets/criteo'))
    # download_criteo(os.path.join(os.path.abspath(__file__), '../datasets/criteo'))
