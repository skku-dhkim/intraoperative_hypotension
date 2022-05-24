import pandas as pd
import glob
import h5py
import pickle
from . import *

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from multiprocessing import Process


def job(file: str, time_seq: int, time_delay: int, prediction_lag: int, dst_path: str) -> None:
    # Get CID
    cid = file.split("/")[-1].split(".")[0]
    # Read csv
    df = pd.read_csv(file)
    # Eliminate NAs
    df = eliminate_na(data_frame=df)
    # Labeling
    df = labeling(input_df=df)
    # Data split
    data_split(cid, df, time_seq=time_seq, time_delay=time_delay, target_seq=prediction_lag, dst_path=dst_path)


def make_dataset(data_path: str, time_seq: int, time_step: int, target_seq: int, dst_path: str) -> None:
    """
    Making dataset from each original case data. Eliminating NA, Fill NA, and labeling are included.
    :param data_path: (str) Original case data directory path
    :param time_seq: (int) Input sequence length. (e.g., 3 minutes to observe for the prediction)
    :param time_step: (int) Steps for time observing sequences.
    :param target_seq: (int) A number of sequences for output target. (e.g., 5 minutes after input)
    :param dst_path: (str) Path for saving result.
    :return: None
    """
    file_list = glob.glob(os.path.join(data_path, "original/*.csv"))
    processes = []
    pbar = tqdm(file_list, desc='Data preprocessing')

    while file_list:
        file = file_list.pop()
        pbar.update(1)
        p = Process(target=job, args=(file, time_seq, time_step, target_seq, dst_path,))
        p.start()
        processes.append(p)

        if len(processes) >= cpu_counts:
            while processes:
                _p = processes.pop()
                _p.join()

    # If process still left, flush...
    if processes:
        while processes:
            _p = processes.pop()
            _p.join()


def data_split(
        cid: str,
        dataframe: DataFrame,
        time_seq: int, time_delay: int, target_seq: int,
        dst_path: str) -> None:
    """
    Data split for train and test.
    Args:
        cid: (str) Case ID
        dataframe: (DataFrame) Pandas DataFrame to process.
        time_seq: (int) Observing sequences.
        time_delay: (int) Time lag.
        target_seq: (int) Prediction target.
        dst_path: (str) Destination path to save file.

    Returns: None

    """
    total_seq = time_seq + target_seq + time_delay
    data_dict = {
        'x': [],
        'y': []
    }

    # NOTE: Make X values and Y values.
    pd_x = dataframe.iloc[:, 1:-1]
    pd_y = dataframe.iloc[:, -1]
    pd_y = pd_y.replace({'normal': 0, 'low': 1, pd.NA: -1})

    # INFO: To numpy
    np_x = pd_x.to_numpy(dtype=np.float16)
    np_y = pd_y.to_numpy(dtype=np.int64)

    for i in range(0, len(np_x) - total_seq, time_delay):
        _x = np_x[i:i + time_seq]
        _y = np_y[i + time_seq + target_seq]

        ############################################################################
        # INFO: Exception mechanism. You may add HERE if another exception needed.
        if np.isnan(_x).any() or _y <= -1:
            continue
        ############################################################################

        data_dict['x'].append(_x)
        data_dict['y'].append(_y)

    # Save numpy using HDF5 format
    date_time = datetime.now().strftime("%Y-%m-%d")

    save_path = os.path.join(dst_path, date_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, "{}.hdf5".format(cid))

    with h5py.File(save_path, "w") as f:
        try:
            f.create_dataset('x', data=data_dict['x'])
            f.create_dataset('y', data=data_dict['y'])
        except Exception as e:
            print(e)
        finally:
            f.close()


def normalization(data_frame: DataFrame, file_save: bool, scaler=None, dataset_path=None):
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
        if os.path.exists(dataset_path + '/sets'):
            os.makedirs(dataset_path + '/sets')
        with open(dataset_path + "/sets/scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)

    return df, scaler


def eliminate_na(data_frame: DataFrame) -> DataFrame:
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

    # NOTE: Sort by Timeline
    _data_frame.sort_values(by=['Time'], inplace=True)

    return _data_frame


def labeling(input_df=None, dataset_path=None) -> DataFrame:
    """
    Labeling the target value
    :param input_df: (Dataframe) Input dataframe
    :param dataset_path: (str) Data path to save
    :return: Dataframe
    """
    if input_df is not None:
        df = input_df.copy(deep=True)
    elif dataset_path is not None:
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("[labeling] Empty parameter.")

    # values = ["low", "high", "normal"]
    values = ["low", "normal", pd.NA]
    conditions = [
        (df['Solar8000/NIBP_SBP'] < 90),
        # (df['Solar8000/NIBP_SBP'] > 180),
        (df['Solar8000/NIBP_SBP'] >= 90),
        (df['Solar8000/NIBP_SBP'].isna())
    ]
    df['Target'] = np.select(conditions, values)

    return df
