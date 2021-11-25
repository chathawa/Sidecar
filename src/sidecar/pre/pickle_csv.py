from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


_DF_LABELS = (
    "date",
    "close",
    "change"
)

_LABEL_SCHEMES = {
    scheme: {
        csv_label: df_label for csv_label, df_label in zip(csv_labels, _DF_LABELS) if csv_label is not None
    }

    for scheme, csv_labels in (
        (
            "yahoo_finance", ("Date", "Close", None)
        ),
    )
}


def prepare_df(df: pd.DataFrame, label_scheme, ticker: str, store_changes=False):
    df.drop(labels=[label for label in df.columns if label not in label_scheme], axis=1, inplace=True)
    df.rename(columns=label_scheme, inplace=True)
    df.insert(
        0,
        "ticker",
        (ticker.upper(),) * df.shape[0]
    )
    df.set_index(["ticker"])

    if store_changes:
        df.insert(
            len(_DF_LABELS) - 1,
            _DF_LABELS[-1],
            [0] + [x2 / x1 for x1, x2 in zip(df['close'][:-1], df['close'][1:])]
        )
        df.drop([0], inplace=True)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "csv_path",
        type=Path
    )
    arg_parser.add_argument(
        "ticker"
    )
    arg_parser.add_argument(
        "out_path",
        type=Path
    )
    arg_parser.add_argument(
        "-l", "--label-scheme",
        choices=list(_LABEL_SCHEMES.keys())
    )
    arg_parser.add_argument(
        "-d", "--store-changes",
        action='store_true',
        default=False
    )
    args = arg_parser.parse_args()

    label_scheme = _LABEL_SCHEMES[args.label_scheme]
    df = pd.read_csv(args.csv_path)
    prepare_df(df, label_scheme, args.ticker, args.store_changes)

    if args.out_path.name.endswith('.pkl'):
        out_path = args.out_path
    else:
        out_path = args.out_path.joinpath(f"{args.csv_path.name.split('.csv')[0]}.pkl")
        args.out_path.mkdir(parents=True, exist_ok=True)

    df.to_pickle(out_path)


if __name__ == '__main__':
    main()
