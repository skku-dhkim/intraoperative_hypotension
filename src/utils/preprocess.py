import pandas as pd
import numpy as np
import glob
import time
import os
import h5py
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from random import sample
from datetime import datetime


def make_dataset(data_path: str,
                 attr: list,
                 time_seq: int = 180,
                 target_seq: int = 300,
                 test_split: float = 0.2,
                 norm='min-max',
                 file_save: bool = True):
    """
    Making dataset from each original case data. Eliminating NA, Fill NA, and labeling are included.
    :param data_path: (str) Original case data directory path
    :param attr: (list) Attributes list
    :param time_seq: (int) Input sequence length. (e.g., 3 minutes to observe for the prediction)
    :param target_seq: (int) A number of sequences for output target. (e.g., 5 minutes after input)
    :param test_split: (float) Train, test proportion.
    :param norm: (bool or None) Normalization method. None if not used.
    :param file_save: (bool) Save everything in file
    :return: Tuple: (Dictionary)train, (Dictionary)test
    """

    if not os.path.exists("{}/labeled_dataset.csv".format(data_path)):
        df = aggregate(data_path, attr=attr)
        df = labeling(input_df=df, dataset_path=data_path, file_save=file_save)
        if norm:
            df = normalization(df, norm, file_save=file_save, dataset_path=data_path)
    else:
        if norm:
            file_path = '{}/normalized_dataset.csv'.format(data_path)
        else:
            file_path = '{}/labeled_dataset.csv'.format(data_path)
        df = pd.read_csv(file_path)

    # NOTE: Train / Test split
    cids = df['CID'].unique()
    n_of_test = int(len(cids)*test_split)
    test_samples = np.array(sample(cids.tolist(), n_of_test))
    test_df = df.loc[df['CID'].isin(test_samples)].reset_index(drop=True)
    train_df = df.loc[~df['CID'].isin(test_samples)].reset_index(drop=True)

    train = data_split(train_df, time_seq=time_seq, target_seq=target_seq, d_type="train", d_path=data_path)
    test = data_split(test_df, time_seq=time_seq, target_seq=target_seq, d_type="test", d_path=data_path)


def data_split(dataframe: pd.DataFrame, time_seq: int, target_seq: int, d_path: str, d_type: str):
    total_seq = time_seq + target_seq
    data_dict = {
        'x': [],
        'y': []
    }

    case_id = dataframe['CID'].unique()

    def _find_sequences(_df):
        # NOTE: Initial Time
        _start_time = _df['Time'][0]

        for idx, _data in _df.iterrows():
            # NOTE: If last value, return others and None.
            if idx+1 >= len(_df):
                return _df[_start_time:], None
            # NOTE: Only get sequential dataframe and others to return.
            if (_df['Time'][idx]+1) != _df['Time'][idx+1]:
                seq = pd.RangeIndex(_start_time, _data.Time+1)
                return _df[_df['Time'].isin(seq)], _df[~_df['Time'].isin(seq)].reset_index(drop=True)

    for cid in tqdm(case_id, desc="Data splitting ({})".format(d_type)):
        tdf = dataframe.loc[dataframe["CID"] == cid].reset_index(drop=True)

        while True:
            sample_df, tdf = _find_sequences(tdf)

            # NOTE: If sub-sampled data sequences has less value than total sequences, continue.
            if len(sample_df) < total_seq:
                if tdf is None:
                    break
                else:
                    continue

            # NOTE: Make X values and Y values.
            pd_x = sample_df.iloc[:, 2:-1]
            pd_y = sample_df.iloc[:, -1]
            pd_y = pd_y.replace({'normal': 0, 'low': 1, 'high': 2})

            # NOTE: To numpy
            np_x = pd_x.to_numpy()
            np_y = pd_y.to_numpy()

            for i in range(len(np_x)-(time_seq+target_seq)-1):
                data_dict['x'].append(np_x[i:i+time_seq])
                data_dict['y'].append(np_y[i+time_seq+target_seq])
            if tdf is None:
                break

    data_dict['x'] = np.array(data_dict['x'])
    data_dict['y'] = np.array(data_dict['y'])

    # Save numpy
    date_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    f = h5py.File("{}/{}_{}.hdf5".format(d_path, d_type, date_time), "w")
    group = f.create_group("{}".format(d_type))
    group.create_dataset("x", data=data_dict['x'])
    group.create_dataset("y", data=data_dict['y'])
    f.close()
    # np.savez_compressed("{}/{}_{}.npz".format(d_path, d_type, date_time), x=data_dict['x'], y=data_dict['y'])
    # np.save("./data/dataset/{}_x_{}.npy".format(d_type, date_time), data_dict['x'])
    # np.save("./data/dataset/{}_y_{}.npy".format(d_type, date_time), data_dict['y'])

    return data_dict


def aggregate(data_path: str, attr: list):
    """
    Aggregate the all the case data.
    :param data_path: (str) dataset path that includes the original directory.
    :param attr: (list) Attributes list
    :return: Dataframe
    """
    df = pd.DataFrame(columns=attr)
    for file_name in tqdm(glob.glob("{}/original/*.csv".format(data_path)), desc="Dataset aggregating"):
        # Get CID
        cid = file_name.split("/")[-1].split(".")[0]
        # 1. Read file and convert into Pandas dataframe
        tdf = pd.read_csv(file_name, header=0)
        tdf['CID'] = cid

        # 2. Eliminate NA
        tdf = eliminate_na(tdf)
        df = df.append(tdf, ignore_index=True)
    return df


def normalization(data_frame: pd.DataFrame, method: str, file_save: bool, dataset_path: str = None):
    """
    Normalize the data.
    :param data_frame: (Dataframe) Pandas input dataframe
    :param method: (str) Normalization method
    :param file_save: (bool) Save or not to save
    :param dataset_path: (dataset_path) Dataset path to save
    :return: Dataframe
    """
    if method == str.lower("min-max"):
        std_scaler = MinMaxScaler()
    elif method == str.lower("std"):
        std_scaler = StandardScaler()
    else:
        raise ValueError("Not supported scale method. Currently supports min-max or std")

    # NOTE: except CID, Time and Target
    _ = std_scaler.fit(data_frame[data_frame.columns[2:-1]])
    np_scaled = std_scaler.transform(data_frame[data_frame.columns[2:-1]])
    np_scaled = pd.DataFrame(np_scaled, columns=data_frame.columns[2:-1], index=list(data_frame.index.values))

    # NOTE: concat CID, Time and Target
    df = pd.concat((data_frame[data_frame.columns[:2]], np_scaled), axis=1)
    df = pd.concat((df, data_frame[data_frame.columns[-1]]), axis=1)

    if file_save:
        if dataset_path is None:
            raise ValueError("You must give dataset_path parameter if file_save is \'True\'")
        # NOTE: Get default dataset path and create save path.
        save_path = dataset_path + "/normalized_dataset.csv"
        df.to_csv(save_path, index=False)

    return df


def eliminate_na(data_frame):
    """
    Eliminate NA on NIBP_SBP and MAC
    :param data_frame: (Dataframe) Input dataframe.
    :return: Dataframe
    """
    # NOTE: Fill front value if NA exists. But not redundantly.
    data_frame.fillna(method="pad", limit=2, inplace=True)

    # NOTE: Extract not NA value from the first to end in Solar8000/NIBP_SBP
    first_idx = data_frame['Solar8000/NIBP_SBP'].first_valid_index()
    last_idx = data_frame['Solar8000/NIBP_SBP'].last_valid_index()
    data_frame = data_frame.loc[first_idx:last_idx]

    # NOTE: Eliminate if NIBP_SBP value still got NA
    data_frame = data_frame[~data_frame['Solar8000/NIBP_SBP'].isnull()]

    # NOTE: Extract not NA value from the first to end in Primus/MAC
    first_idx = data_frame['Primus/MAC'].first_valid_index()
    last_idx = data_frame['Primus/MAC'].last_valid_index()
    data_frame['Primus/MAC'].fillna(method="pad", inplace=True)
    data_frame = data_frame.loc[first_idx:last_idx]

    # NOTE: Sort by Timeline
    data_frame.sort_values(by=['Time'], inplace=True)

    return data_frame


def labeling(input_df=None, dataset_path=None, file_save=False):
    """
    Labeling the target value
    :param input_df: (Dataframe) Input dataframe
    :param dataset_path: (str) Data path to save
    :param file_save: (bool) To save file or not to save.
    :return: Dataframe
    """
    if input_df is not None:
        df = input_df.copy(deep=True)
    elif dataset_path is not None:
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("[labeling] Empty parameter.")

    values = ["low", "high", "normal"]
    conditions = [
        (df['Solar8000/NIBP_SBP'] < 90),
        (df['Solar8000/NIBP_SBP'] > 180),
        (df['Solar8000/NIBP_SBP'] >= 90) & (df['Solar8000/NIBP_SBP'] <= 180)
    ]
    df['Target'] = np.select(conditions, values)
    if file_save:
        # NOTE: Get default dataset path and create save path.
        save_path = dataset_path + "/labeled_dataset.csv"
        df.to_csv(save_path, index=False)
    return df
