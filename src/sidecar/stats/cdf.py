from argparse import ArgumentParser
from pathlib import Path

from scipy.stats import norm
import pandas as pd
import numpy as np


def x_steps(y: np.ndarray, num_steps: int):
    start, stop = min(y), max(y)
    return np.arange(start, stop, (stop - start) / num_steps)


def ecdf(x: np.ndarray, y: np.ndarray):
    return np.array([
        np.where(y <= value)[0].shape[0] / x.shape[0] for value in x
    ])


def cdf(x: np.ndarray, y: np.ndarray, scale=1.0):
    return scale * norm.cdf(x, loc=y.mean(), scale=y.std())


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "df_path",
        type=Path
    )
    arg_parser.add_argument(
        "out_path",
        type=Path
    )
    arg_parser.add_argument(
        "-c", "--columns",
        choices=["close", "change"],
        default=["change"]
    )
    arg_parser.add_argument(
        "-n", "--num-steps",
        type=int,
        default=100
    )
    args = arg_parser.parse_args()

    df = pd.read_pickle(args.df_path)
    data = {}
    for label in args.columns:
        column = df[label]
        y = column if label == 'close' else np.log(column)
        x = x_steps(y, args.num_steps)
        ecdf_y = ecdf(x, y)

        data[label] = {
            "ecdf": ecdf_y,
            "cdf": cdf(x, y, max(ecdf_y))
        }

    if args.out_path.name.endswith('.pkl'):
        out_path = args.out_path
    else:
        out_path = args.out_path.joinpath(f"{args.df_path.name.split('.pkl')[0]}_cdf.pkl")
        args.out_path.mkdir(parents=True, exist_ok=True)