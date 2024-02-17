import enum
import sys
import os

ctr_dir = os.path.realpath(os.path.dirname(__file__))
root_dir = os.path.realpath(os.path.join(ctr_dir, os.pardir, os.pardir))
sys.path.append(os.path.join(root_dir, "python"))
sys.path.append(os.path.join(root_dir, "build", "lib"))
sys.path.append(os.path.join(root_dir, "third_party", "GraphMix", "python"))
sys.path.append(os.path.join(root_dir, "third_party", "HetuML", "hetuml", "python"))

# enable nccl
# os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "eno0"
# os.environ["PS_VERBOSE"] = "2"
os.environ["DMLC_INTERFACE"] = "eno0"

import hetu as ht

import os.path as osp
import numpy as np
import pandas as pd
import time
import argparse
from tqdm import tqdm
from sklearn import metrics


def worker(args):
    def train(iterations, auc_enabled=True, tqdm_enabled=False, log_file=None):
        # print("start train")
        localiter = tqdm(range(iterations)) if tqdm_enabled else range(iterations)
        train_loss = []
        train_acc = []
        if auc_enabled:
            train_auc = []
        time_collector = []
        i = 0
        # print("to enter train loop")
        for it in localiter:
            i = i + 1
            # print("begin to monitor time")
            start_time = time.time()
            loss_val, predict_y, y_val, _ = executor.run(
                "train", convert_to_numpy_ret_vals=True
            )
            if y_val.shape[1] == 1:  # for criteo case
                acc_val = np.equal(y_val, predict_y > 0.5).astype(np.float32)
            else:
                acc_val = np.equal(np.argmax(y_val, 1), np.argmax(predict_y, 1)).astype(
                    np.float32
                )
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
            if auc_enabled:
                train_auc.append(metrics.roc_auc_score(y_val, predict_y))
            end_time = time.time()
            # print("end of monitor time: {}".format(end_time - start_time))
            execution_time = end_time - start_time
            time_collector.append(execution_time)
        time_collector = np.array(time_collector)

        if log_file != None:
            log_file.write(
                "\nTotal number of mini-batch is {}\n".format(len(time_collector))
            )
            log_file.write("Total time is {}\n".format(np.sum(time_collector)))
            log_file.write("Average time is {}\n".format(np.mean(time_collector)))
            log_file.write("Max time is {}\n".format(np.max(time_collector)))
            log_file.write("Min time is {}\n".format(np.min(time_collector)))
            log_file.write("Epoch time is: {} epochs\n".format(len(time_collector)))
            for delta in time_collector:
                log_file.write("{}\n".format(delta))
            log_file.flush()
        if auc_enabled:
            return np.mean(train_loss), np.mean(train_acc), np.mean(train_auc)
        else:
            return np.mean(train_loss), np.mean(train_acc)

    def validate(iterations, tqdm_enabled=False):
        localiter = tqdm(range(iterations)) if tqdm_enabled else range(iterations)
        test_loss = []
        test_acc = []
        test_auc = []
        for it in localiter:
            loss_val, test_y_predicted, y_test_val = executor.run(
                "validate", convert_to_numpy_ret_vals=True
            )
            if y_test_val.shape[1] == 1:  # for criteo case
                correct_prediction = np.equal(
                    y_test_val, test_y_predicted > 0.5
                ).astype(np.float32)
            else:
                correct_prediction = np.equal(
                    np.argmax(y_test_val, 1), np.argmax(test_y_predicted, 1)
                ).astype(np.float32)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            test_auc.append(metrics.roc_auc_score(y_test_val, test_y_predicted))
        return np.mean(test_loss), np.mean(test_acc), np.mean(test_auc)

    def get_current_shard(data):
        return data

    batch_size = args.batch_size
    dataset = args.dataset
    model = args.model
    embedding_size = args.embedding_size
    cache_ratio = args.cache_limit_ratio
    print(f"cache limit ratio: {args.cache_limit_ratio}")
    cache_limit = 0

    model_name = args.raw_model.split('_')[0]
    if model_name == 'fae':
        if dataset == 'criteo':
            # define models for criteo
            if args.all:
                from models.load_data import process_all_criteo_data_in_fae_mode
                dense, hot_sparse, cold_sparse, cold_category, labels = process_all_criteo_data_in_fae_mode(
                    return_val=args.val)
            else:
                raise NotImplementedError
            if isinstance(dense, tuple):
                dense_input = ht.dataloader_op([[dense[0], batch_size, 'train'], [
                                            dense[1], batch_size, 'validate']])
                hot_sparse_input = ht.dataloader_op([[hot_sparse[0], batch_size, 'train'], [
                                                hot_sparse[1], batch_size, 'validate']])
                cold_sparse_input = ht.dataloader_op([[cold_sparse[0], batch_size, 'train'], [
                                                cold_sparse[1], batch_size, 'validate']])
                cold_category_input = ht.dataloader_op([[cold_category[0], batch_size, 'train'], [
                                                cold_category[1], batch_size, 'validate']])
                y_ = ht.dataloader_op([[labels[0], batch_size, 'train'], [
                                    labels[1], batch_size, 'validate']])
            else:
                dense_input = ht.dataloader_op(
                    [[dense, batch_size, 'train']])
                hot_sparse_input = ht.dataloader_op(
                    [[hot_sparse, batch_size, 'train']])
                cold_sparse_input = ht.dataloader_op(
                    [[cold_sparse, batch_size, 'train']])
                cold_category_input = ht.dataloader_op(
                    [[cold_category, batch_size, 'train']])
                y_ = ht.dataloader_op(
                    [[labels, batch_size, 'train']])
        elif dataset == 'avazu':
            # define models for criteo
            if args.all:
                from models.load_data import process_all_avazu_data_in_fae_mode
                dense, hot_sparse, cold_sparse, cold_category, labels = process_all_avazu_data_in_fae_mode(
                    return_val=args.val)
            else:
                raise NotImplementedError
            if isinstance(dense, tuple):
                dense_input = ht.dataloader_op([[dense[0], batch_size, 'train'], [
                                            dense[1], batch_size, 'validate']])
                hot_sparse_input = ht.dataloader_op([[hot_sparse[0], batch_size, 'train'], [
                                                hot_sparse[1], batch_size, 'validate']])
                cold_sparse_input = ht.dataloader_op([[cold_sparse[0], batch_size, 'train'], [
                                                cold_sparse[1], batch_size, 'validate']])
                cold_category_input = ht.dataloader_op([[cold_category[0], batch_size, 'train'], [
                                                cold_category[1], batch_size, 'validate']])
                y_ = ht.dataloader_op([[labels[0], batch_size, 'train'], [
                                    labels[1], batch_size, 'validate']])
            else:
                dense_input = ht.dataloader_op(
                    [[dense, batch_size, 'train']])
                hot_sparse_input = ht.dataloader_op(
                    [[hot_sparse, batch_size, 'train']])
                cold_sparse_input = ht.dataloader_op(
                    [[cold_sparse, batch_size, 'train']])
                cold_category_input = ht.dataloader_op(
                    [[cold_category, batch_size, 'train']])
                y_ = ht.dataloader_op(
                    [[labels, batch_size, 'train']])
        elif dataset == 'criteosearch':
            # define models for criteo
            if args.all:
                from models.load_data import process_all_criteo_search_data_in_fae_mode
                dense, hot_sparse, cold_sparse, cold_category, labels = process_all_criteo_search_data_in_fae_mode(
                    return_val=args.val)
            else:
                raise NotImplementedError
            if isinstance(dense, tuple):
                dense_input = ht.dataloader_op([[dense[0], batch_size, 'train'], [
                                            dense[1], batch_size, 'validate']])
                hot_sparse_input = ht.dataloader_op([[hot_sparse[0], batch_size, 'train'], [
                                                hot_sparse[1], batch_size, 'validate']])
                cold_sparse_input = ht.dataloader_op([[cold_sparse[0], batch_size, 'train'], [
                                                cold_sparse[1], batch_size, 'validate']])
                cold_category_input = ht.dataloader_op([[cold_category[0], batch_size, 'train'], [
                                                cold_category[1], batch_size, 'validate']])
                y_ = ht.dataloader_op([[labels[0], batch_size, 'train'], [
                                    labels[1], batch_size, 'validate']])
            else:
                dense_input = ht.dataloader_op(
                    [[dense, batch_size, 'train']])
                hot_sparse_input = ht.dataloader_op(
                    [[hot_sparse, batch_size, 'train']])
                cold_sparse_input = ht.dataloader_op(
                    [[cold_sparse, batch_size, 'train']])
                cold_category_input = ht.dataloader_op(
                    [[cold_category, batch_size, 'train']])
                y_ = ht.dataloader_op(
                    [[labels, batch_size, 'train']])
        elif dataset == 'movie':
            # define models for criteo
            if args.all:
                from models.load_data import process_all_movie_data_in_fae_mode
                trainData, hot_sparse_user, cold_sparse_user, hot_sparse_item, cold_sparse_item, cold_user_category, cold_item_category = process_all_movie_data_in_fae_mode()
                trainLabels = get_current_shard(trainData['labels']).reshape(-1,1)
            else:
                raise NotImplementedError
            hot_sparse_user_input = ht.dataloader_op([
                ht.Dataloader(hot_sparse_user, batch_size, 'train'),
            ])
            cold_sparse_user_input = ht.dataloader_op([
                ht.Dataloader(cold_sparse_user, batch_size, 'train'),
            ])
            hot_sparse_item_input = ht.dataloader_op([
                ht.Dataloader(hot_sparse_item, batch_size, 'train'),
            ])
            cold_sparse_item_input = ht.dataloader_op([
                ht.Dataloader(cold_sparse_item, batch_size, 'train'),
            ])
            cold_user_category_input = ht.dataloader_op([
                ht.Dataloader(cold_user_category, batch_size, 'train'),
            ])
            cold_item_category_input = ht.dataloader_op([
                ht.Dataloader(cold_item_category, batch_size, 'train'),
            ])
            y_ = ht.dataloader_op([
                ht.Dataloader(trainLabels, batch_size, 'train')
            ])

        else:
            raise NotImplementedError
        cache_limit = 500
    else:
        if dataset == "criteo":
            # define models for criteo
            if args.all:
                from models.load_data import process_all_criteo_data

                dense, sparse, labels = process_all_criteo_data(return_val=args.val)
            elif args.val:
                from models.load_data import process_head_criteo_data

                dense, sparse, labels = process_head_criteo_data(return_val=True)
            else:
                from models.load_data import process_sampled_criteo_data

                dense, sparse, labels = process_sampled_criteo_data()

            # calculate cache limit
            cache_limit = int(args.cache_limit_ratio * np.amax(sparse)) + 1
            if isinstance(dense, tuple):
                dense_input = ht.dataloader_op(
                    [
                        [dense[0], batch_size, "train"],
                        [dense[1], batch_size, "validate"],
                    ]
                )
                sparse_input = ht.dataloader_op(
                    # sparse_input = ht.dataloader_with_push_index_op(
                    [
                        [sparse[0], batch_size, "train"],
                        [sparse[1], batch_size, "validate"],
                    ]
                )
                y_ = ht.dataloader_op(
                    [
                        [labels[0], batch_size, "train"],
                        [labels[1], batch_size, "validate"],
                    ]
                )
            else:
                dense_input = ht.dataloader_op([[dense, batch_size, "train"]])
                sparse_input = ht.dataloader_op(
                    # sparse_input = ht.dataloader_with_push_index_op(
                    [[sparse, batch_size, "train"]]
                )
                y_ = ht.dataloader_op([[labels, batch_size, "train"]])
        elif dataset == "criteosearch":
            # define models for criteo
            if args.all:
                from models.load_data import process_all_criteo_search_data

                dense, sparse, labels = process_all_criteo_search_data(
                    return_val=args.val
                )
            else:
                raise PermissionError("Wrong arg for criteo search dataset")

            # calculate cache limit
            cache_limit = int(args.cache_limit_ratio * np.amax(sparse)) + 1
            if isinstance(dense, tuple):
                dense_input = ht.dataloader_op(
                    [
                        [dense[0], batch_size, "train"],
                        [dense[1], batch_size, "validate"],
                    ]
                )
                sparse_input = ht.dataloader_op(
                    [
                        [sparse[0], batch_size, "train"],
                        [sparse[1], batch_size, "validate"],
                    ]
                )
                y_ = ht.dataloader_op(
                    [
                        [labels[0], batch_size, "train"],
                        [labels[1], batch_size, "validate"],
                    ]
                )
            else:
                dense_input = ht.dataloader_op([[dense, batch_size, "train"]])
                sparse_input = ht.dataloader_op([[sparse, batch_size, "train"]])
                y_ = ht.dataloader_op([[labels, batch_size, "train"]])
        elif dataset == "avazu":
            # define models for criteo
            if args.all:
                from models.load_data import process_all_avazu_data

                dense, sparse, labels = process_all_avazu_data(return_val=args.val)
            else:
                raise PermissionError("Wrong arg for avazu dataset")

            # calculate cache limit
            cache_limit = args.cache_limit_ratio * np.amax(sparse) + 1

            if isinstance(dense, tuple):
                dense_input = ht.dataloader_op(
                    [
                        [dense[0], batch_size, "train"],
                        [dense[1], batch_size, "validate"],
                    ]
                )
                sparse_input = ht.dataloader_op(
                    [
                        [sparse[0], batch_size, "train"],
                        [sparse[1], batch_size, "validate"],
                    ]
                )
                y_ = ht.dataloader_op(
                    [
                        [labels[0], batch_size, "train"],
                        [labels[1], batch_size, "validate"],
                    ]
                )
            else:
                dense_input = ht.dataloader_op([[dense, batch_size, "train"]])
                sparse_input = ht.dataloader_op([[sparse, batch_size, "train"]])
                y_ = ht.dataloader_op([[labels, batch_size, "train"]])
        elif dataset == "adult":
            from models.load_data import load_adult_data

            (
                x_train_deep,
                x_train_wide,
                y_train,
                x_test_deep,
                x_test_wide,
                y_test,
            ) = load_adult_data()
            dense_input = [
                ht.dataloader_op(
                    [
                        [x_train_deep[:, i], batch_size, "train"],
                        [x_test_deep[:, i], batch_size, "validate"],
                    ]
                )
                for i in range(12)
            ]
            sparse_input = ht.dataloader_op(
                [
                    [x_train_wide, batch_size, "train"],
                    [x_test_wide, batch_size, "validate"],
                ]
            )
            y_ = ht.dataloader_op(
                [
                    [y_train, batch_size, "train"],
                    [y_test, batch_size, "validate"],
                ]
            )
        elif dataset == "movie":
            from models.load_data import process_all_movie_data

            if args.all:
                sparse, labels = process_all_movie_data(return_val=args.val)
            else:
                raise PermissionError("Wrong arg for movie dataset")
            cache_limit = int(args.cache_limit_ratio * np.amax(sparse)) + 1
            if args.val:
                sparse_input = ht.dataloader_op(
                    [
                        ht.Dataloader(sparse[0], batch_size, "train"),
                        ht.Dataloader(sparse[1], batch_size, "validate"),
                    ]
                )
                y_ = ht.dataloader_op(
                    [
                        ht.Dataloader(labels[0], batch_size, "train"),
                        ht.Dataloader(labels[1], batch_size, "validate"),
                    ]
                )
            else:
                sparse_input = ht.dataloader_op(
                    [ht.Dataloader(sparse, batch_size, "train")]
                )
                y_ = ht.dataloader_op([ht.Dataloader(labels, batch_size, "train")])
        else:
            raise NotImplementedError
    print("Data loaded.")

    cache_perf_enable = args.cache_perf
    if cache_perf_enable:
        print("Hetu: enable cache performance monitor")

    if model_name == 'fae':
        if dataset == 'movie':
            loss, prediction, y_, train_op = model(hot_sparse_user_input, cold_sparse_user_input, hot_sparse_item_input, cold_sparse_item_input, cold_user_category_input, cold_item_category_input, y_, embedding_size)
        else:
            loss, prediction, y_, train_op = model(dense_input, hot_sparse_input, cold_sparse_input, cold_category_input, y_, embedding_size)
    else:
        if dataset == 'movie':
            loss, prediction, y_, train_op = model(sparse_input, y_, embedding_size)
        else:
            loss, prediction, y_, train_op = model(dense_input, sparse_input, y_, embedding_size)

    eval_nodes = {"train": [loss, prediction, y_, train_op]}
    if args.val and dataset != "movie":
        print("Validation enabled...")
        eval_nodes["validate"] = [loss, prediction, y_]
    executor_log_path = osp.join(osp.dirname(osp.abspath(__file__)), "logs")
    strategy = ht.dist.DataParallel(aggregate=args.comm)
    # print("Hetu: eval_nodes: {}".format(eval_nodes))
    print(f"cache limit: {cache_limit}")
    executor = ht.Executor(
        eval_nodes,
        dist_strategy=strategy,
        cstable_policy=args.cache,
        bsp=args.bsp,
        cache_bound=args.bound,
        cache_limit=cache_limit,
        cache_perf_enable=cache_perf_enable,
        seed=123,
        log_path=executor_log_path,
        # prefetch=False,
    )

    if args.all and (
        dataset == "criteo"
        or dataset == "avazu"
        or dataset == "criteosearch"
        or dataset == "movie"
    ):
        print("Processing all data...")
        file_path = "%s_%s" % (
            {None: "local", "PS": "ps", "Hybrid": "hybrid", "AllReduce": "allreduce"}[
                args.comm
            ],
            args.raw_model,
        )
        file_path += "%d.log" % executor.rank if args.comm else ".log"
        file_path = osp.join(osp.dirname(osp.abspath(__file__)), "logs", file_path)
        log_file = open(file_path, "w")
        total_epoch = args.nepoch if args.nepoch > 0 else 11
        for ep in range(total_epoch):
            print("ep: %d" % ep)
            ep_st = time.time()
            # train_loss, train_acc, train_auc = train(executor.get_batch_num(
            #     'train') // 10 + (ep % 10 == 9) * (executor.get_batch_num('train') % 10), tqdm_enabled=True)
            train_auc = 0.0
            train_loss, train_acc = train(
                executor.get_batch_num("train"),
                tqdm_enabled=True,
                auc_enabled=False,
                log_file=log_file,
            )
            ep_en = time.time()
            if args.val and dataset != 'movie' and model_name != 'fae':
                val_loss, val_acc, val_auc = validate(
                    executor.get_batch_num("validate")
                )
                printstr = (
                    "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"
                    % (
                        train_loss,
                        train_acc,
                        train_auc,
                        val_loss,
                        val_acc,
                        val_auc,
                        ep_en - ep_st,
                    )
                )
            else:
                printstr = (
                    "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"
                    % (train_loss, train_acc, train_auc, ep_en - ep_st)
                )
            print(printstr)
            log_file.write(printstr + "\n")
            log_file.flush()
        # if executor.rank == 0:
        caches = train_op.get_cache()
        for idx, cache in enumerate(caches):
            if cache is not None:
                csv_file_path = "hetu_cache%d_%d.csv" % (idx, executor.rank)
                csv_file_path = osp.join(
                    osp.dirname(osp.abspath(__file__)), "csv", csv_file_path
                )
                pd.DataFrame(cache.get_perf()).to_csv(csv_file_path)
            else:
                print("Cache perf is None")
    else:
        total_epoch = args.nepoch if args.nepoch > 0 else 50
        start = time.time()
        for ep in range(total_epoch):
            if ep == 5:
                start = time.time()
            print("epoch %d" % ep)
            ep_st = time.time()
            train_loss, train_acc = train(
                executor.get_batch_num("train"), auc_enabled=False
            )
            ep_en = time.time()
            if args.val:
                val_loss, val_acc, val_auc = validate(
                    executor.get_batch_num("validate")
                )
                print(
                    "train_loss: %.4f, train_acc: %.4f, train_time: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f"
                    % (train_loss, train_acc, ep_en - ep_st, val_loss, val_acc, val_auc)
                )
            else:
                print(
                    "train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                    % (train_loss, train_acc, ep_en - ep_st)
                )
        print("all time:", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to be tested")
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "-e", "--embedding-size", type=int, default=128, help="embedding size"
    )
    parser.add_argument(
        "-r",
        "--cache-limit-ratio",
        type=float,
        default=0.1,
        help="ratio of cache limit to total embedding sizes",
    )
    parser.add_argument(
        "--cache-perf",
        action="store_true",
        help="whether to enable cache performance measurement",
    )
    parser.add_argument("--val", action="store_true", help="whether to use validation")
    parser.add_argument("--all", action="store_true", help="whether to use all data")
    parser.add_argument(
        "--comm",
        default=None,
        help="whether to use distributed setting, can be None, AllReduce, PS, Hybrid",
    )
    parser.add_argument("--bsp", type=int, default=-1, help="bsp 0, asp -1, ssp > 0")
    parser.add_argument("--cache", default=None, help="cache policy")
    parser.add_argument("--bound", default=100, help="cache bound")
    parser.add_argument(
        "--nepoch", type=int, default=-1, help="num of epochs, each train 1/10 data"
    )
    args = parser.parse_args()
    import models

    print("Model:", args.model)
    model = eval("models." + args.model)
    args.dataset = args.model.split("_")[-1]
    args.raw_model = args.model
    args.model = model
    worker(args)
