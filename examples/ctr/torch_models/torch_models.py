import torch, os
import torch.nn as nn
import prettytable as pt
import numpy as np
import copy
from DeepLight.model.DeepFMs import DeepFMs
from DeepLight.NFM import NFM
from DeepLight.utils.data_preprocess import read_data
from PyTorch_GBW_LM.lm.util import initialize
from PyTorch_GBW_LM.lm.model import SampledSoftmax, RNNModel
from PyTorch_GBW_LM.lm.stream_gbw import Vocabulary, StreamGBWDataset

# feature_dimension = 33762577
# learning_rate = 0.01

list_mini_batch = [8, 16, 128, 256]
list_dimension = [128, 256, 512]

feature_size_criteo = {8:148.4121, 16:264.0573, 32:473.7479, 64:853.4464, 128:1533.7634, 256:2736.3720, 512:4822.7359}
feature_size_avazu = {8:80.67984, 16:134.56514, 32:226.4950, 64:380.0111, 128:629.76776, 256:1030.4909, 512:1675.6945}
feature_size_lm = {8:115.2904, 16:209.6096, 128:1207.5619, 256:2098.2371}
user_size_movie = {8:7.9993, 16:15.9970, 32: 31.9875, 64:63.9494, 128:127.7973, 256:255.1911, 512:508.7941}
item_size_movie = {8:7.9989, 16:15.9953, 32: 31.9807, 64:63.9217, 128:127.6858, 256:254.7493, 512:507.0450}
feature_size_criteosearch = {32: 204.0023, 64: 358.9886, 128: 639.3451, 256: 1146.8726, 512: 2068.1567}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids = [0, 1]
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

class wdl_criteo(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(wdl_criteo, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_criteo[mini_batch])
        self.embedding = torch.nn.Embedding(self.feature_dimension, self.embedding_size)
        self.layer1 = nn.Sequential(
            nn.Linear(13, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256, bias = False)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256 + 26 * self.embedding_size, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, dense_input, sparse_input):
        dense_input = dense_input.to(torch.float32)
        # sparse_input = sparse_input.to(torch.float32)
        y1 = self.layer1(dense_input)
        y2 = self.embedding(sparse_input)
        y2 = torch.reshape(y2, (-1, 26 * self.embedding_size))
        y3 = torch.concat((y1, y2), axis=1)
        y4 = self.layer2(y3)
        return y4

class dfm_criteo(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(dfm_criteo, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_criteo[mini_batch])
        self.embedding1 = torch.nn.Embedding(self.feature_dimension, 1)
        self.embedding2 = torch.nn.Embedding(self.feature_dimension, self.embedding_size)

        self.fm_layer1 = nn.Linear(13, 1, bias = False)

        self.layer1 = nn.Sequential(
            nn.Linear(26 * self.embedding_size, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1, bias = False)
        )
        self.activation = nn.Sigmoid()

    def forward(self, dense_input, sparse_input):
        sparse_1dim_input = self.embedding1(sparse_input)
        fm_dense_part = self.fm_layer1(dense_input)
        fm_sparse_part = torch.sum(sparse_1dim_input, dim = 1)
        y1 = fm_dense_part + fm_sparse_part

        sparse_2dim_input = self.embedding2(sparse_input)
        sparse_2dim_sum = torch.sum(sparse_2dim_input, dim = 1)
        sparse_2dim_sum_square = torch.square(sparse_2dim_sum)

        sparse_2dim_square = torch.square(sparse_2dim_input)
        sparse_2dim_square_sum = torch.sum(sparse_2dim_square, dim = 1)
        sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
        sparse_2dim_half = sparse_2dim * 0.5

        y2 = torch.sum(sparse_2dim_half, dim = 1, keepdim = True)

        flatten = torch.reshape(sparse_2dim_input, (-1, 26 * self.embedding_size))
        y3 = self.layer1(flatten)

        y4 = y1 + y2
        y = y4 + y3
        y = self.activation(y)

        return y

class dfm_avazu(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(dfm_avazu, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_avazu[mini_batch])
        self.embedding1 = torch.nn.Embedding(self.feature_dimension, 1)
        self.embedding2 = torch.nn.Embedding(self.feature_dimension, self.embedding_size)

        self.fm_layer1 = nn.Linear(4, 1, bias = False)

        self.layer1 = nn.Sequential(
            nn.Linear(18 * self.embedding_size, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1, bias = False)
        )
        self.activation = nn.Sigmoid()

    def forward(self, dense_input, sparse_input):
        sparse_1dim_input = self.embedding1(sparse_input)
        fm_dense_part = self.fm_layer1(dense_input)
        fm_sparse_part = torch.sum(sparse_1dim_input, dim = 1)
        y1 = fm_dense_part + fm_sparse_part

        sparse_2dim_input = self.embedding2(sparse_input)
        sparse_2dim_sum = torch.sum(sparse_2dim_input, dim = 1)
        sparse_2dim_sum_square = torch.square(sparse_2dim_sum)

        sparse_2dim_square = torch.square(sparse_2dim_input)
        sparse_2dim_square_sum = torch.sum(sparse_2dim_square, dim = 1)
        sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
        sparse_2dim_half = sparse_2dim * 0.5

        y2 = torch.sum(sparse_2dim_half, dim = 1, keepdim = True)

        flatten = torch.reshape(sparse_2dim_input, (-1, 26 * self.embedding_size))
        y3 = self.layer1(flatten)

        y4 = y1 + y2
        y = y4 + y3
        y = self.activation(y)

        return y

class dcn_criteo(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(dcn_criteo, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_criteo[mini_batch])
        self.embedding = torch.nn.Embedding(self.feature_dimension, self.embedding_size)

        self.embedding_len = 26 * self.embedding_size + 13
        self.crosslayer1 = nn.Linear(self.embedding_len, 1, bias = False)
        self.crosslayer2 = nn.Linear(self.embedding_len, 1, bias = False)
        self.crosslayer3 = nn.Linear(self.embedding_len, 1, bias = False)

        self.bias1 = nn.Parameter(torch.randn(self.embedding_len, 1))
        self.bias2 = nn.Parameter(torch.randn(self.embedding_len, 1))
        self.bias3 = nn.Parameter(torch.randn(self.embedding_len, 1))
        # for i in range(3):
        #     self.crosslayer.append(nn.Linear(26 * 128 + 13, 1, bias = True))

        self.layer1 = nn.Sequential(
            nn.Linear(self.embedding_len, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256, bias = False)
        )

        self.layer2 = nn.Linear(256 + self.embedding_len, 1, bias = False)
        self.layer3 = nn.Sigmoid()
        

    def forward(self, dense_input, sparse_input):
        new_sparse_input = self.embedding(sparse_input)
        new_sparse_input = torch.reshape(new_sparse_input, (-1, 26 * self.embedding_size))
        x = torch.concat(new_sparse_input, dense_input, axis=1)

        x1w = self.crosslayer1(x)
        y = torch.mul(x, torch.broadcast_to(x1w, x.shape))
        y = y + x + torch.broadcast_to(self.bias1, y.shape)
        x_previous = y

        x1w = self.crosslayer2(x_previous)
        y = torch.mul(x, torch.broadcast_to(x1w, x.shape))
        y = y + x_previous + torch.broadcast_to(self.bias2, y.shape)
        x_previous = y

        x1w = self.crosslayer3(x_previous)
        y = torch.mul(x, torch.broadcast_to(x1w, x.shape))
        cross_output = y + x_previous + torch.broadcast_to(self.bias3, y.shape)

        y3 = self.layer1(x)
        y4 = torch.concat(cross_output, y3, axis = 1)

        y = self.layer2(y4)
        y = self.layer3(y)

        return y

class dcn_criteosearch(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(dcn_criteosearch, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 17
        self.feature_dimension = round(feature_size_criteosearch[mini_batch])
        self.embedding = torch.nn.Embedding(self.feature_dimension, self.embedding_size)

        self.embedding_len = 17 * self.embedding_size + 3
        self.crosslayer1 = nn.Linear(self.embedding_len, 1, bias = False)
        self.crosslayer2 = nn.Linear(self.embedding_len, 1, bias = False)
        self.crosslayer3 = nn.Linear(self.embedding_len, 1, bias = False)

        self.bias1 = nn.Parameter(torch.randn(self.embedding_len, 1))
        self.bias2 = nn.Parameter(torch.randn(self.embedding_len, 1))
        self.bias3 = nn.Parameter(torch.randn(self.embedding_len, 1))
        # for i in range(3):
        #     self.crosslayer.append(nn.Linear(17 * 128 + 3, 1, bias = True))

        self.layer1 = nn.Sequential(
            nn.Linear(self.embedding_len, 64, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(64, 32, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(32, 16, bias = False)
        )

        self.layer2 = nn.Linear(16 + self.embedding_len, 1, bias = False)
        self.layer3 = nn.Sigmoid()
        

    def forward(self, dense_input, sparse_input):
        new_sparse_input = self.embedding(sparse_input)
        new_sparse_input = torch.reshape(new_sparse_input, (-1, 17 * self.embedding_size))
        x = torch.concat(new_sparse_input, dense_input, axis=1)

        x1w = self.crosslayer1(x)
        y = torch.mul(x, torch.broadcast_to(x1w, x.shape))
        y = y + x + torch.broadcast_to(self.bias1, y.shape)
        x_previous = y

        x1w = self.crosslayer2(x_previous)
        y = torch.mul(x, torch.broadcast_to(x1w, x.shape))
        y = y + x_previous + torch.broadcast_to(self.bias2, y.shape)
        x_previous = y

        x1w = self.crosslayer3(x_previous)
        y = torch.mul(x, torch.broadcast_to(x1w, x.shape))
        cross_output = y + x_previous + torch.broadcast_to(self.bias3, y.shape)

        y3 = self.layer1(x)
        y4 = torch.concat(cross_output, y3, axis = 1)

        y = self.layer2(y4)
        y = self.layer3(y)

        return y

class ncf(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(ncf, self).__init__()
        self.layers = [64, 32, 16, 8]
        self.embedding_size = embedding_size
        self.embedding_len = self.embedding_size + self.layers[0] // 2
        self.User_Embedding = torch.nn.Embedding(round(user_size_movie[mini_batch]), self.embedding_len)
        self.Item_Embedding = torch.nn.Embedding(round(item_size_movie[mini_batch]), self.embedding_len)
        self.layer1 = nn.Sequential(
            nn.Linear(self.layers[0], self.layers[1], bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(self.layers[1], self.layers[2], bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(self.layers[2], self.layers[3], bias = False),
            nn.ReLU(inplace = True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.layers[3] + self.embedding_size, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, user_input, item_input):
        user_latent = self.User_Embedding(user_input)
        item_latent = self.Item_Embedding(item_input)
        mf_user_latent = user_latent[:, 0:self.embedding_size]
        mlp_user_latent = user_latent[:, self.embedding_size:self.embedding_len]
        mf_item_latent = item_latent[:, 0:self.embedding_size]
        mlp_item_latent = item_latent[:, self.embedding_size:self.embedding_len]
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)
        mlp_vector = torch.concat(mlp_user_latent, mlp_item_latent, axis=1)
        mlp_output = self.layer1(mlp_vector)
        concat_vector = torch.concat(mf_vector, mlp_output, axis=1)
        y = self.layer2(concat_vector)

        return y

class rmc2_criteo(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(rmc2_criteo, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_criteo[mini_batch])
        self.embedding = torch.nn.Embedding(self.feature_dimension, self.embedding_size)
        self.bottomlayer = nn.Sequential(
            nn.Linear(13, 512, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 64, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(64, self.embedding_size, bias = False),
        )
        self.toplayer = nn.Sequential(
            nn.Linear(self.embedding_size + 26 * self.embedding_size, 512, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, dense_input, sparse_input):
        y1 = self.bottomlayer(dense_input)
        y2 = self.embedding(sparse_input)
        y2 = torch.reshape(y2, (-1, 26 * self.embedding_size))
        # Refered from https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py
        (batch_size, d) = y1.shape
        T = torch.cat([y1] + y2, dim=1).view((batch_size, -1, d))
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        # approach 1: all
        # Zflat = Z.view((batch_size, -1))
        # approach 2: unique
        _, ni, nj = Z.shape
        # approach 1: tril_indices
        # offset = 0 if self.arch_interaction_itself else -1
        # li, lj = torch.tril_indices(ni, nj, offset=offset)
        # approach 2: custom
        offset = 1 if self.arch_interaction_itself else 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]
        # concatenate dense features and interactions
        R = torch.cat([y1] + [Zflat], dim=1)

        y = self.toplayer(R)
        return y

class rmc4_avazu(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(rmc4_avazu, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 21
        self.feature_dimension = round(feature_size_avazu[mini_batch])
        self.embedding = torch.nn.Embedding(self.feature_dimension, self.embedding_size)
        self.bottomlayer = nn.Sequential(
            nn.Linear(1, 512, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 64, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(64, self.embedding_size, bias = False),
        )
        self.toplayer = nn.Sequential(
            nn.Linear(self.embedding_size + 21 * self.embedding_size, 512, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, dense_input, sparse_input):
        y1 = self.bottomlayer(dense_input)
        y2 = self.embedding(sparse_input)
        y2 = torch.reshape(y2, (-1, 21 * self.embedding_size))
        # Refered from https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py
        (batch_size, d) = y1.shape
        T = torch.cat([y1] + y2, dim=1).view((batch_size, -1, d))
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        # approach 1: all
        # Zflat = Z.view((batch_size, -1))
        # approach 2: unique
        _, ni, nj = Z.shape
        # approach 1: tril_indices
        # offset = 0 if self.arch_interaction_itself else -1
        # li, lj = torch.tril_indices(ni, nj, offset=offset)
        # approach 2: custom
        offset = 1 if self.arch_interaction_itself else 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]
        # concatenate dense features and interactions
        R = torch.cat([y1] + [Zflat], dim=1)

        y = self.toplayer(R)
        return y

class deeplight_criteo(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(deeplight_criteo, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_criteosearch[mini_batch])
        self.embedding1 = torch.nn.Embedding(self.feature_dimension, 1)
        self.embedding2 = torch.nn.Embedding(self.feature_dimension, self.embedding_size)

        self.fm_layer1 = nn.Linear(3, 1, bias = True)

        self.activation = nn.Sigmoid()

    def forward(self, dense_input, sparse_input):
        sparse_1dim_input = self.embedding1(sparse_input)
        fm_dense_part = self.fm_layer1(dense_input)
        fm_sparse_part = torch.sum(sparse_1dim_input, dim = 1)
        y1 = fm_dense_part + fm_sparse_part

        sparse_2dim_input = self.embedding2(sparse_input)
        sparse_2dim_sum = torch.sum(sparse_2dim_input, dim = 1)
        sparse_2dim_sum_square = torch.square(sparse_2dim_sum)

        sparse_2dim_square = torch.square(sparse_2dim_input)
        sparse_2dim_square_sum = torch.sum(sparse_2dim_square, dim = 1)
        sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
        sparse_2dim_half = sparse_2dim * 0.5

        y2 = torch.sum(sparse_2dim_half, dim = 1, keepdim = True)

        y = y1 + y2
        y = self.activation(y)

        return y

class deeplightlr_avazu(nn.Module):
    def __init__(self, mini_batch, embedding_size):
        super(deeplightlr_avazu, self).__init__()
        self.embedding_size = embedding_size
        # self.feature_dimension = mini_batch * 26
        self.feature_dimension = round(feature_size_avazu[mini_batch])
        self.embedding = torch.nn.Embedding(self.feature_dimension, 1)

        self.fm_layer1 = nn.Linear(4, 1, bias = True)

        self.activation = nn.Sigmoid()

    def forward(self, dense_input, sparse_input):
        sparse_1dim_input = self.embedding1(sparse_input)
        fm_dense_part = self.fm_layer1(dense_input)
        fm_sparse_part = torch.sum(sparse_1dim_input, dim = 1)
        y = fm_dense_part + fm_sparse_part

        y = self.activation(y)

        return y

def wdl_criteo_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = wdl_criteo(mini_batch, dimension)
            # print (parameter_num)
            dense_para = 0
            embedding_para = 0
            # print([p.numel() for p in model.parameters() if p.requires_grad])
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("wdl_criteo")
    print(table)
    table.clear_rows()

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

def criteo_lookup_size():
    dense, sparse, labels = process_all_criteo_data(return_val=False)
    dataset = torch.utils.data.TensorDataset(torch.tensor(dense), torch.tensor(sparse), torch.tensor(labels))
    
    print(dense.shape, sparse.shape, labels.shape)
    for mini_batch in list_mini_batch:
        data_iter = torch.utils.data.DataLoader(dataset, mini_batch, shuffle=True)
        print("Dataloader done for mini_batch")
        num_batch = sum(1 for _ in data_iter)
        print("Number of batches is ", num_batch)
        num_category = 0
        for _, sparse_input, _ in data_iter:
            feature_dict = []
            for i in range(26):
                feature_dict.append({})
            for example in sparse_input:
                # print(example)
                for i in range(26):
                    # print(example[i].item())
                    feature_dict[i][example[i].item()] = 1
                    # print(feature_dict[i])
            # print(feature_dict)
            num_list = [len(feature_dict[i]) for i in range(26)]
            num_category += sum(num_list)
        print("Average number of feature categories in each batch is ", 1.0 * num_category / num_batch)

def process_all_avazu_data(path=os.path.join(os.path.split(os.path.abspath(__file__))[0], '../datasets/avazu'), return_val=True):
    file_paths = [os.path.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]

    files = [np.load(filename, mmap_mode='r').reshape(-1,1) if 'labels' in filename else np.load(filename, mmap_mode='r') for filename in file_paths]
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]

def avazu_lookup_size():
    dense, sparse, labels = process_all_avazu_data(return_val=False)
    dataset = torch.utils.data.TensorDataset(torch.tensor(dense), torch.tensor(sparse), torch.tensor(labels))
    
    print(dense.shape, sparse.shape, labels.shape)
    for mini_batch in list_mini_batch:
        data_iter = torch.utils.data.DataLoader(dataset, mini_batch, shuffle=True)
        print("Dataloader done for mini_batch")
        num_batch = sum(1 for _ in data_iter)
        print("Number of batches is ", num_batch)
        num_category = 0
        for _, sparse_input, _ in data_iter:
            feature_dict = []
            for i in range(18):
                feature_dict.append({})
            for example in sparse_input:
                # print(example)
                for i in range(18):
                    # print(example[i].item())
                    feature_dict[i][example[i].item()] = 1
                    # print(feature_dict[i])
            # print(feature_dict)
            num_list = [len(feature_dict[i]) for i in range(18)]
            num_category += sum(num_list)
        print("Average number of feature categories in each batch is ", 1.0 * num_category / num_batch)

def gbw_lookup_size():
    data_path = "../datasets/gbw"
    vocabulary = Vocabulary.from_file(os.path.join("./PyTorch_GBW_LM/data/lm1b", "1b_word_vocab.txt"))

    train_corpus = StreamGBWDataset(vocabulary, os.path.join(data_path, "training-monolingual.tokenized.shuffled/*"))
    # test_corpus = StreamGBWDataset(vocabulary, os.path.join(data_path, "heldout-monolingual.tokenized.shuffled/*"), deterministic=True)

    for mini_batch in list_mini_batch:
        train_loader = train_corpus.batch_generator(seq_length = 20, batch_size = mini_batch)
        avg_token = 0
        num_batch = 0
        for data, _, _ in train_loader:
            num_batch = num_batch + 1
            # print(data.shape)
            feature_dict = {}
            for index in data:
                for sample in index:
                    feature_dict[sample.item()] = 1
            num_token = len(feature_dict)
            # print(num_token)
            avg_token += num_token
            # feature_dict[i][example[i].item()] = 1
        print("Num of batches is ", num_batch)
        print("Average number of different tokens in each batch is ", 1.0 * avg_token / num_batch)

def criteo_kaggle_lookup_size():
    criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    result_dict = read_data('./DeepLight/data/tiny_train_input.csv', './DeepLight/data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
    
    print(dense.shape, sparse.shape, labels.shape)
    for mini_batch in list_mini_batch:
        data_iter = torch.utils.data.DataLoader(dataset, mini_batch)
        print("Dataloader done for mini_batch")
        num_batch = sum(1 for _ in data_iter)
        print("Number of batches is ", num_batch)
        num_category = 0
        for _, sparse_input, _ in data_iter:
            feature_dict = []
            for i in range(18):
                feature_dict.append({})
            for example in sparse_input:
                # print(example)
                for i in range(18):
                    # print(example[i].item())
                    feature_dict[i][example[i].item()] = 1
                    # print(feature_dict[i])
            # print(feature_dict)
            num_list = [len(feature_dict[i]) for i in range(18)]
            num_category += sum(num_list)
        print("Average number of feature categories in each batch is ", 1.0 * num_category / num_batch)
  
def dfm_criteo_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = dfm_criteo(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("deepfm_criteo")
    print(table)
    table.clear_rows()

def dfm_avazu_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = dfm_avazu(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("dfm_avazu")
    print(table)
    table.clear_rows()

def dcn_criteo_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = dcn_criteo(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("dcn_criteo")
    print(table)
    table.clear_rows()

def dcn_criteosearch_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = dcn_criteosearch(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("dcn_critesearch")
    print(table)
    table.clear_rows()

def process_all_movie_data(dataset='ml-25m', data_dir='../../rec/datasets/'):
    data_subdir = os.path.join(data_dir, dataset)
    file_paths = [os.path.join(data_subdir, data)
                  for data in ['train.npz', 'test.npy']]
    return np.load(file_paths[0])

def movie_lookup_size():
    train_data = process_all_movie_data()
    trainUsers = train_data['user_input']
    trainItems = train_data['item_input']
    trainLabels = train_data['labels']
    dataset = torch.utils.data.TensorDataset(torch.tensor(trainUsers), torch.tensor(trainItems), torch.tensor(trainLabels))
    
    for mini_batch in list_mini_batch:
        data_iter = torch.utils.data.DataLoader(dataset, mini_batch, shuffle=True)
        # print("Dataloader done for mini_batch")
        num_batch = 0
        num_user = 0
        num_item = 0
        for user_input, item_input, _ in data_iter:
            num_batch = num_batch + 1
            user_dict = {}
            item_dict = {}
            for users in user_input:
                user_dict[users.item()] = 1
            for items in item_input:
                item_dict[items.item()] = 1

            num_user += len(user_dict)
            num_item += len(item_dict)
        print("Number of batches is ", num_batch)
        print("Average number of users and items in each batch are seperately ", 1.0 * num_user / num_batch, 1.0 * num_item / num_batch)

def ncf_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = ncf(mini_batch, dimension)
            # print ([p.numel() for p in model.parameters()])
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'Embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("ncf")
    print(table)
    table.clear_rows()

def rmc2_criteo_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = rmc2_criteo(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("rmc2_criteo")
    print(table)
    table.clear_rows()

def rmc4_avazu_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = rmc4_avazu(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("rmc4_avazu")
    print(table)
    table.clear_rows()

def deeplight_criteo_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = deeplight_criteo(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("deeplight_criteosearch")
    print(table)
    table.clear_rows()

def deeplightlr_avazu_param(table):
    for mini_batch in list_mini_batch:
        for dimension in list_dimension:
            model = deeplightlr_avazu(mini_batch, dimension)
            dense_para = 0
            embedding_para = 0
            for name, para in model.named_parameters():
                if 'embedding' in name:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("deeplightlr_avazu")
    print(table)
    table.clear_rows()

def LM_param(table):
    twht = None
    word_freq = torch.load('./PyTorch_GBW_LM/data/lm1b/word_freq.pt').numpy()
    # print(len(word_freq))
    # ntokens = len(word_freq)
    for mini_batch in list_mini_batch:
        # ntokens = mini_batch
        ntokens = round(feature_size_lm[mini_batch])
        for dimension in list_dimension:
            D = dimension
            ss = SampledSoftmax(ntokens, 8192, D, tied_weight=twht)
            model = RNNModel(ntokens, dimension, 2048, dimension, 1, True, 0.01)
            encoder = nn.Embedding(ntokens, dimension, sparse=True)
            initialize(encoder.weight)
            model.add_module("encoder", encoder)
            model.add_module("decoder", ss)
            # print (parameter_num)
            index = 0
            dense_para = 0
            embedding_para = 0
            # print([p.numel() for p in model.parameters() if p.requires_grad])
            for para in model.parameters():
                if index == 6:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
                index += 1
            # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    print("LM")
    print(table)
    table.clear_rows()

def para_cal(model):
    dense_para = 0
    embedding_para = 0
    
    for name, param in model.named_parameters():
        # print (name, param.data.shape)
        # num_total += np.prod(param.data.shape)
        if 'embeddings' in name:
            param_index = int(name.strip().split('.')[1])
            if param_index < 13:
                dense_para += np.prod(param.data.shape)
            else:
                embedding_para += np.prod(param.data.shape)
        else:
            dense_para += np.prod(param.data.shape)
    return dense_para, embedding_para

def DeepFMs_param(table):
    table_lr = copy.deepcopy(table)
    table_fm = copy.deepcopy(table)
    table_fwfm = copy.deepcopy(table)
    table_deepfm = copy.deepcopy(table)
    table_nfm = copy.deepcopy(table)

    criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # result_dict = read_data('./DeepLight/data/tiny_train_input.csv', './DeepLight/data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
    result_dict = read_data('./DeepLight/data/large/train.csv', './DeepLight/data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
    print("feature size is ", result_dict['feature_sizes'], "sum is ", np.sum(np.array(result_dict['feature_sizes'])))

    dataset = torch.utils.data.TensorDataset(torch.tensor(result_dict['value']), torch.tensor(result_dict['index']), torch.tensor(result_dict['label']))
    
    for mini_batch in list_mini_batch:
        
        data_iter = torch.utils.data.DataLoader(dataset, mini_batch)
        # print("Dataloader done for mini_batch")
        num_batch = sum(1 for _ in data_iter)
        print("Number of batches is ", num_batch)

        num_array = np.zeros([26, ])
        
        for _, sparse_input, _ in data_iter:
            feature_dict = []
            for i in range(26):
                feature_dict.append({})
            for example in sparse_input:
                # print(example)
                for i in range(26):
                    # print(example[i].item())
                    feature_dict[i][example[i].item()] = 1
                    # print(feature_dict[i])
            # print(feature_dict)
            num_list = [len(feature_dict[i]) for i in range(26)]
            num_array = num_array + np.array(num_list)
        num_array = num_array // num_batch
        feature_sizes = [1] * 13
        feature_sizes = feature_sizes + list(num_array.astype(np.int))
        print("feature_sizes is ", feature_sizes)
        
        # feature_sizes=[mini_batch * 26]
        for dimension in list_dimension:
            model = DeepFMs(field_size=39, feature_sizes = feature_sizes, embedding_size=dimension, n_epochs=8, \
                verbose=True, use_cuda=1, use_fm=0, use_fwfm=0, use_ffm=0, use_deep=0, \
                batch_size=mini_batch, learning_rate=0.001, weight_decay=3e-7, momentum=0, sparse=0.9, warm=10, \
                h_depth=3, deep_nodes=400, num_deeps=1, numerical=13, use_lw=0, use_fwlw=0, \
                use_logit=1, random_seed=0) # LR
            dense_para, embedding_para = para_cal(model)
            table_lr.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])

            model = DeepFMs(field_size=39, feature_sizes = feature_sizes, embedding_size=dimension, n_epochs=8, \
                verbose=True, use_cuda=1, use_fm=1, use_fwfm=0, use_ffm=0, use_deep=0, \
                batch_size=mini_batch, learning_rate=0.001, weight_decay=3e-7, momentum=0, sparse=0.9, warm=10, \
                h_depth=3, deep_nodes=400, num_deeps=1, numerical=13, use_lw=0, use_fwlw=0, \
                use_logit=0, random_seed=0) # FM
            dense_para, embedding_para = para_cal(model)
            table_fm.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])

            model = DeepFMs(field_size=39, feature_sizes = feature_sizes, embedding_size=dimension, n_epochs=8, \
                verbose=True, use_cuda=1, use_fm=0, use_fwfm=1, use_ffm=0, use_deep=0, \
                batch_size=mini_batch, learning_rate=0.001, weight_decay=3e-7, momentum=0, sparse=0.9, warm=10, \
                h_depth=3, deep_nodes=400, num_deeps=1, numerical=13, use_lw=0, use_fwlw=0, \
                use_logit=0, random_seed=0) # FwFM
            dense_para, embedding_para = para_cal(model)
            table_fwfm.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])

            model = DeepFMs(field_size=39, feature_sizes = feature_sizes, embedding_size=dimension, n_epochs=8, \
                verbose=True, use_cuda=1, use_fm=1, use_fwfm=0, use_ffm=0, use_deep=1, \
                batch_size=mini_batch, learning_rate=0.001, weight_decay=3e-7, momentum=0, sparse=0.9, warm=10, \
                h_depth=3, deep_nodes=400, num_deeps=1, numerical=13, use_lw=0, use_fwlw=0, \
                use_logit=0, random_seed=0) # DeepFM
            dense_para, embedding_para = para_cal(model)
            table_deepfm.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])

            model = NFM(field_size=39, feature_sizes = feature_sizes, embedding_size = dimension, batch_size=mini_batch, is_shallow_dropout=False, verbose=True, use_cuda=True,
                      weight_decay=3e-7, use_fm=True, use_ffm=False, interation_type=False, learning_rate=1e-3) # NFM
            dense_para, embedding_para = para_cal(model)
            table_nfm.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])

    print("DeepFMs_LR")
    print(table_lr)
    print("DeepFMs_FM")
    print(table_fm)
    print("DeepFMs_FwFM")
    print(table_fwfm)
    print("DeepFMs_FwFM")
    print(table_fwfm)
    print("DeepFMs_DeepFM")
    print(table_deepfm)
    print("DeepFMs_NFM")
    print(table_nfm)
    table.clear_rows()

def DeepFMs_param_with_pruning(table):
    criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    result_dict = read_data('./DeepLight/data/tiny_train_input.csv', './DeepLight/data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
    test_dict = read_data('./DeepLight/data/tiny_test_input.csv', './DeepLight/data/category_emb', criteo_num_feat_dim, feature_dim_start=0, dim=39)
    # result_dict = read_data('./DeepLight/data/large/train.csv', './DeepLight/data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
    # test_dict = read_data('./DeepLight/data/large/valid.csv', './DeepLight/data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
    # print("feature size is ", result_dict['feature_sizes'], "sum is ", np.sum(np.array(result_dict['feature_sizes'])))
    feature_sizes=result_dict['feature_sizes']
    print(feature_sizes)
    for mini_batch in list_mini_batch[2:3]:
        # feature_sizes=[mini_batch * 26]
        for dimension in list_dimension[0:1]:
            model = DeepFMs(field_size=39, feature_sizes = feature_sizes, embedding_size=dimension, n_epochs=8, \
                verbose=True, use_cuda=1, use_fm=1, use_fwfm=0, use_ffm=0, use_deep=1, \
                batch_size=mini_batch, learning_rate=0.001, weight_decay=3e-7, momentum=0, sparse=0.9, warm=10, \
                h_depth=3, deep_nodes=400, num_deeps=1, numerical=13, use_lw=0, use_fwlw=0, \
                use_logit=0, random_seed=0) # Without pruning
            # model = DeepFMs(field_size=39, feature_sizes = feature_sizes, embedding_size=dimension, n_epochs=5, \
            #     verbose=True, use_cuda=1, use_fm=0, use_fwfm=1, use_ffm=0, use_deep=1, \
            #     batch_size=mini_batch, learning_rate=0.001, weight_decay=6e-7, momentum=0, sparse=0.9, warm=2, \
            #     h_depth=3, deep_nodes=400, num_deeps=1, numerical=13, use_lw=1, use_fwlw=1, \
            #     use_logit=0, random_seed=0) # With pruning
            model = model.cuda()
            print([p.numel() for p in model.parameters() if p.requires_grad])
            print([p.numel() for name, p in model.named_parameters()])
            save_model_name = './DeepLight/saved_models/' + 'DeepFwFM' + '_l2_' + str(6e-7) + '_sparse_' + str(0.90) + '_seed_' + str(0)
            model.fit(result_dict['index'], result_dict['value'], result_dict['label'], test_dict['index'], test_dict['value'], test_dict['label'], \
                prune=1, prune_fm=1, prune_r=1, prune_deep=1, save_path=save_model_name, emb_r=0.444, emb_corr=1.0)

            # print (parameter_num)
            index = 0
            dense_para = 0
            embedding_para = 0
            for para in model.parameters():
                if index == 1 or index == 2:
                    embedding_para += para.numel()
                else:
                    if para.requires_grad:
                        dense_para += para.numel()
                index += 1
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            table.add_row([mini_batch, dimension, dense_para, embedding_para, embedding_para / (embedding_para + dense_para)])
    # print("DeepFMs")
    # print(table)
    table.clear_rows()

if __name__=="__main__":
    table = pt.PrettyTable()
    table.field_names = ["mini-batch", "dimension", "Dense paras", "Embedding paras", "Embedding paras / Total paras"]
    
    # criteo_lookup_size()
    # avazu_lookup_size()
    # gbw_lookup_size()
    # movie_lookup_size()
    # dfm_criteo_param(table)
    # dcn_criteo_param(table)
    wdl_criteo_param(table)
    ncf_param(table)
    dfm_avazu_param(table)
    dcn_criteosearch_param(table)
    # rmc2_criteo_param(table)
    # rmc4_avazu_param(table)
    # LM_param(table)
    # DeepFMs_param(table) # Runtime is about 4 hours.
    # deeplight_criteo_param(table)
    # deeplightlr_avazu_param(table)