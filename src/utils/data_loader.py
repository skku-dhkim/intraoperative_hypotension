from src.vitaldb_framework import vitaldb
import pandas as pd
import numpy as np
import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from typing import Callable


def data_load(data_path, attr, maxcases):
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    _attributes = attr[2:]
    _ = vitaldb.load_cases(
        tnames=_attributes,
        path_for_save="{}/original".format(data_path),
        interval=1,
        maxcases=maxcases
    )


def composite(data_path, attr):
    df = pd.DataFrame(columns=attr)
    total = len(glob.glob("./data/{}/original/*.csv".format(data_path)))
    count = 0
    for file_name in glob.glob("./data/{}/original/*.csv".format(data_path)):
        cid = file_name.split("/")[-1].split(".")[0]
        print("Running on : {}".format(cid))
        tdf = pd.read_csv(file_name, header=0)
        tdf['CID'] = cid
        df = df.append(tdf, ignore_index=True)
        print("Job: {}/{}".format(count, total))
        count += 1

    print("Saving...")
    df.to_csv("./data/{}/dataset.csv".format(data_path), index=False)
    print("[Done] Saving...")
    return "./data/{}/dataset.csv".format(data_path)


def matching_caseID(data_path):
    dataset = pd.read_csv("./data/{}/dataset.csv".format(data_path))
    case_id = dataset['CID'].unique()

    try:
        case_info = pd.read_csv("./data/total_cases.csv".format(data_path))
        case_info['caseid'] = case_info['caseid'].apply(pd.to_numeric)

        case_info = case_info[case_info['caseid'].isin(case_id)]
        case_info.to_csv("./data/{}/case_info.csv".format(data_path), index=False)
        print("[Done] Matching IDs...")
    except FileNotFoundError:
        raise FileNotFoundError("You should move or create \'total_cases.csv\' file first.")


class VitalDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(Dataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        x = np.array(self.x[index], dtype=np.float32)
        y = np.array(self.y[index], dtype=np.int64)
        return x, y

    def __len__(self):
        return len(self.x)

    def get_labels(self):
        return list(self.y)


# class CustomWeightedRandomSampler(WeightedRandomSampler):
#     """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __iter__(self):
#         rand_tensor = np.random.choice(range(0, len(self.weights)),
#                                        size=self.num_samples,
#                                        p=self.weights.numpy() / torch.sum(self.weights).numpy(),
#                                        replace=self.replacement)
#         rand_tensor = torch.from_numpy(rand_tensor)
#         return iter(rand_tensor.tolist())


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):

        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=True)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


def load_from_hdf5(file_path):
    f = h5py.File(file_path, 'r')

    data_x = f['train']['x']
    data_y = f['train']['y']

    return data_x, data_y
