from distutils.util import strtobool

import src.utils.data_loader as dl
import src.utils.preprocess as prep
import argparse
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_load", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--file_save", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--save_sql", type=lambda x: bool(strtobool(x)), default=False)

    # parser.add_argument("--data_make", type=lambda x: bool(strtobool(x)), default=False)

    parser.add_argument("--dset_path", type=str, required=True)
    parser.add_argument("--time_seq", type=int, default=30)
    parser.add_argument("--target_seq", type=int, default=300)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--test_split", type=float, default=0.2)
    # parser.add_argument("--norm", type=str, default='std')

    parser.add_argument("--maxcase", type=int, default=-1)

    args = parser.parse_args()
    if args.save_sql:
        prep.save_sql(args.dset_path, args.test_split)
        print("[Done] Save SQL.")
        raise SystemExit

    attributes = ["CID",
                  "Time",
                  "SNUADC/ECG_II",
                  "Solar8000/NIBP_SBP",
                  "Solar8000/NIBP_MBP",
                  "Solar8000/NIBP_DBP",
                  "SNUADC/PLETH",
                  "Primus/MAC",
                  "Primus/CO2",
                  "BIS/BIS"]
    if args.data_load:
        # 1. load from vital DB framework
        dl.data_load(args.dset_path, attributes, args.maxcase, args.interval)
        prep.make_dataset(
            args.dset_path,
            attr=attributes,
            file_save=True,
            # norm=args.norm,
            time_seq=args.time_seq,
            target_seq=args.target_seq,
            test_split=args.test_split
        )
    else:
        prep.make_dataset(
            args.dset_path,
            attr=attributes,
            file_save=args.file_save,
            sql=args.save_sql,
            # norm=args.norm,
            time_seq=args.time_seq,
            target_seq=args.target_seq,
            test_split=args.test_split
        )

    print("[Done] Data maker.")
