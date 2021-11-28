from argparse import ArgumentParser
from pathlib import Path

from sidecar.cdf import CDFDataset, x_steps, ecdf, cdf
from sidecar.datasets import *
from sidecar.prices import PricesDataset




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