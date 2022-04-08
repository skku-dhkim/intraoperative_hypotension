import multiprocessing

import pandas as pd
import numpy as np
import glob
import time
import os
import h5py
import pickle
import itertools
import psycopg2 as db
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from random import sample
from datetime import datetime
from multiprocessing import Process, current_process, Pool, RLock, freeze_support
from sqlalchemy import create_engine


def save_sql(data_path: str, test_split: float):
    original_files = glob.glob("{}/original/*.csv".format(data_path))
    n_of_train = int(len(original_files)*(1-test_split))
    train_samples = np.array(sample(original_files, n_of_train))
    aggregate(train_samples, dtype='train')
    print("Work done #1")

    np_original = np.array(original_files)
    test_samples = np.delete(np_original, [np.where(np_original == train_sample) for train_sample in train_samples])
    aggregate(test_samples, dtype='test')
    print("Work done #2")


def make_dataset(data_path: str,
                 attr: list,
                 time_seq: int = 180,
                 target_seq: int = 300,
                 test_split: float = 0.2,
                 file_save: bool = False,
                 sql: bool = True):
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
    db_engine = create_engine('postgresql://aai:aai2022@221.163.101.162:5432/aihub')
    train_original = pd.read_sql("SELECT * FROM vital.\"train.eliminated\"", db_engine)

    # NOTE: Normalization
    # train_df, scaler = normalization(train_df, file_save=True, dataset_path=data_path)
    # test_df, _ = normalization(test_df, scaler=scaler, file_save=False)
    # print("Normalization Done...")
    #
    # number_of_files = 10
    # cids = train_df['CID'].unique()
    # process_per_case = int(len(cids)/number_of_files)
    # jobs = []
    # freeze_support()
    # pool = Pool(processes=55, initargs=(RLock(), ), initializer=tqdm.set_lock)
    #
    # for i in range(number_of_files):
    #     if (i+1)*process_per_case > len(cids):
    #         cid_list = cids[i * process_per_case:]
    #     else:
    #         cid_list = cids[i*process_per_case: (i+1)*process_per_case]
    #     tdf = train_df.loc[train_df["CID"].isin(cid_list)].reset_index(drop=True)
    #     jobs.append(pool.apply_async(data_split, args=(tdf, time_seq, target_seq, data_path, 'train', i)))
    # [job.get() for job in jobs]
    # pool.close()
    # pool.join()
    # print('\n'*(number_of_files+1))

    # if norm:
    #     if os.path.exists("{}/normalized_dataset.csv".format(data_path)):
    #         train_df = pd.read_csv("{}/normalized_dataset.csv".format(data_path))
    #         test_df = pd.read_csv("{}/normalized_test_dataset.csv".format(data_path))
    #     else:
    #         train_df, scaler = normalization(train_df, norm, file_save=file_save, dataset_path=data_path)
    #         # NOTE: except CID, Time and Target
    #         np_scaled = scaler.transform(test_df[test_df.columns[2:-1]])
    #         np_scaled = pd.DataFrame(np_scaled, columns=test_df.columns[2:-1], index=list(test_df.index.values))
    #
    #         # NOTE: concat CID, Time and Target
    #         np_scaled = pd.concat((test_df[test_df.columns[:2]], np_scaled), axis=1)
    #         test_df = pd.concat((np_scaled, test_df[test_df.columns[-1]]), axis=1)
    #
    #         if file_save:
    #             if data_path is None:
    #                 raise ValueError("You must give dataset_path parameter if file_save is \'True\'")
    #             # NOTE: Get default dataset path and create save path.
    #             save_path = data_path + "/normalized_test_dataset.csv"
    #             test_df.to_csv(save_path, index=False)
    #
    # train = data_split(train_df, time_seq=time_seq, target_seq=target_seq, d_type="train", d_path=data_path)
    # test = data_split(test_df, time_seq=time_seq, target_seq=target_seq, d_type="test", d_path=data_path)


def data_split(dataframe: pd.DataFrame, time_seq: int, time_delay: int, target_seq: int, d_path: str, d_type: str, pid: int):
    total_seq = time_seq + target_seq + time_delay
    data_dict = {
        'x': [],
        'y': []
    }

    case_id = dataframe['CID'].unique()

    # def _find_sequences(_df):
    #     # NOTE: Initial Time
    #     _start_time = _df['Time'][0]
    #
    #     for idx, _data in _df.iterrows():
    #         # NOTE: If last value, return others and None.
    #         if idx+1 >= len(_df):
    #             return _df[_start_time:], None
    #         # NOTE: Only get sequential dataframe and others to return.
    #         if (_df['Time'][idx]+1) != _df['Time'][idx+1]:
    #             seq = pd.RangeIndex(_start_time, _data.Time+1)
    #             return _df[_df['Time'].isin(seq)], _df[~_df['Time'].isin(seq)].reset_index(drop=True)

    for cid in tqdm(case_id, desc="#{} Data splitting ({})".format(pid, d_type)):
        tdf = dataframe.loc[dataframe["CID"] == cid].reset_index(drop=True)
        # while True:
        #     # sample_df, tdf = _find_sequences(tdf)
        #
        #     # # NOTE: If sub-sampled data sequences has less value than total sequences, continue.
        #     # if len(sample_df) < total_seq:
        #     #     if tdf is None:
        #     #         break
        #     #     else:
        #     #         continue

        # NOTE: Make X values and Y values.
        pd_x = tdf.iloc[:, 2:-1]
        pd_y = tdf.iloc[:, -1]
        pd_y = pd_y.replace({'normal': 0, 'low': 1})

        # NOTE: To numpy
        np_x = pd_x.to_numpy()
        np_y = pd_y.to_numpy()

        for i in range(len(np_x)-total_seq+1):
            data_dict['x'].append(np_x[i*time_delay:(i*time_delay)+time_seq])
            data_dict['y'].append(np_y[(i*time_delay)+time_seq+target_seq])

    data_dict['x'] = np.array(data_dict['x'])
    data_dict['y'] = np.array(data_dict['y'])

    # Save numpy using HDF5 fomat
    date_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    if not os.path.exists("{}/sets"):
        os.makedirs("{}/sets_{}".format(d_path, date_time))
    f = h5py.File("{}/sets/{}_part{}.hdf5".format(d_path, d_type, pid), "w")
    group = f.create_group("{}".format(d_type))
    group.create_dataset("x", data=data_dict['x'])
    group.create_dataset("y", data=data_dict['y'])
    f.close()


def csv_to_pandas(pid, file_list, dtype='train'):
    """
    Read CSV file from original data path.
        1. Convert them into Pandas Dataframe.
        2. Add CID into Dataframe
        3. Eliminate NAs
        4. Labeling
        5. Save into Postgres DB
    Args:
        pid: Process ID
        file_list: List of file to handle.
        dtype: Type of data. It is used for table name of postgres DB.

    Returns:

    """
    with tqdm(file_list, desc="#{}".format(pid), position=pid+1) as pbar:
        for file_name in file_list:
            # Get CID
            cid = file_name.split("/")[-1].split(".")[0]
            # 1. Read file and convert into Pandas dataframe
            tdf = pd.read_csv(file_name, header=0)
            tdf['CID'] = cid

            # 2. Eliminate NA
            tdf = eliminate_na(tdf)

            # 3. Labeling
            tdf = labeling(input_df=tdf, file_save=False)

            # 4. To database
            db_engine = create_engine('postgresql://aai:aai2022@221.163.101.162:5432/aihub')
            tdf.to_sql('{}.eliminated'.format(dtype), db_engine,
                       schema='vital',
                       if_exists="append",
                       method='multi',
                       chunksize=1000,
                       index=False)
            pbar.update(1)


def aggregate(file_list, dtype='train'):
    """
    Aggregates the original files and then eliminates NA and Labeling, and save into Postgres.
    Multiprocessing is used for large dataset to handle.

    Args:
        file_list: List of files to process.
        dtype: Data type value. It will be used for table name.

    Returns:
        None.
    """
    freeze_support()
    files_per_process = int((len(file_list) / multiprocessing.cpu_count())) + 1
    pool = Pool(processes=multiprocessing.cpu_count(), initargs=(RLock(), ), initializer=tqdm.set_lock)
    jobs = []
    for process_id in range(multiprocessing.cpu_count()):
        if (process_id+1)*files_per_process > len(file_list):
            _file_list = file_list[process_id * files_per_process:]
        else:
            _file_list = file_list[process_id*files_per_process:(process_id+1)*files_per_process]
        jobs += [pool.apply_async(csv_to_pandas, args=(process_id, _file_list, dtype, ))]
    [job.get() for job in jobs]     # Nothing to get, but call get method just in case.
    pool.close()
    pool.join()


def normalization(data_frame: pd.DataFrame, file_save: bool, scaler=None, dataset_path=None):
    """
    Normalize the data.
    :param data_frame: (Dataframe) Pandas input dataframe
    :param file_save: (bool) Save or not to save
    :param scaler: (StandardScaler()) StandardScaler from scikit-learn
    :param dataset_path: (str) Dataset path
    :return: Dataframe, scaler
    """
    if scaler is None:
        scaler = StandardScaler()

    # NOTE: except CID, Time and Target
    _ = scaler.fit(data_frame[data_frame.columns[2:-1]])
    np_scaled = scaler.transform(data_frame[data_frame.columns[2:-1]])
    np_scaled = pd.DataFrame(np_scaled, columns=data_frame.columns[2:-1], index=list(data_frame.index.values))

    # NOTE: concat CID, Time and Target
    df = pd.concat((data_frame[data_frame.columns[:2]], np_scaled), axis=1)
    df = pd.concat((df, data_frame[data_frame.columns[-1]]), axis=1)

    if file_save:
        if os.path.exists(dataset_path+'/sets'):
            os.makedirs(dataset_path+'/sets')
        with open(dataset_path + "/sets/scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)

    return df, scaler


def eliminate_na(data_frame):
    """
    Eliminate NA on NIBP_SBP and MAC
    :param data_frame: (Dataframe) Input dataframe.
    :return: Dataframe
    """
    _data_frame = data_frame.copy(deep=True)

    # NOTE: 1. Extract valid range which Solar8000/NIBP_SBP value exist.
    first_idx = _data_frame['Solar8000/NIBP_SBP'].first_valid_index()
    last_idx = _data_frame['Solar8000/NIBP_SBP'].last_valid_index()
    _data_frame = _data_frame.loc[first_idx:last_idx]

    # NOTE: 2. Fill the value if NA exists. But not redundantly 300 times.
    _data_frame.fillna(method="ffill", inplace=True)

    # NOTE: Eliminate if NIBP_SBP value still got NA
    _data_frame = _data_frame.dropna()

    # NOTE: Sort by Timeline
    _data_frame.sort_values(by=['Time'], inplace=True)

    return _data_frame


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

    # values = ["low", "high", "normal"]
    values = ["low", "normal"]
    conditions = [
        (df['Solar8000/NIBP_SBP'] < 90),
        # (df['Solar8000/NIBP_SBP'] > 180),
        (df['Solar8000/NIBP_SBP'] >= 90)
    ]
    df['Target'] = np.select(conditions, values)
    if file_save:
        # NOTE: Get default dataset path and create save path.
        save_path = dataset_path + "/labeled_dataset.csv"
        df.to_csv(save_path, index=False)
    return df
