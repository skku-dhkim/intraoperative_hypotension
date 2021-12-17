from src.vitaldb_framework import vitaldb
import pandas as pd
import numpy as np
import os
import glob


def data_load(data_path, attr, maxcases):
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    _attributes = attr[2:]
    _ = vitaldb.load_cases(
        tnames=_attributes,
        path_for_save="./data/{}/original".format(data_path),
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
    # return "./data/{}/case_info.csv".format(data_path)
