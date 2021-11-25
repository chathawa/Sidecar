from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm


def plot_dist(df: pd.DataFrame, *columns, image_path=None):
    fig, axes = plt.subplots(1, len(columns))

    if len(columns) == 1:
        axes = (axes,)

    ticker = df['ticker'].values[0]
    start_date, end_date = min(df['date']), max(df['date'])
    for label, ax in zip(columns, axes):
        y = np.log(df[label].to_numpy())

        start, stop, num_steps = min(y), max(y), 100
        x = np.arange(start, stop, (stop - start) / num_steps)

        ecdf = np.array([np.where(y <= value)[0].shape[0] / x.shape[0] for value in x])
        ax.plot(x, ecdf, color='red')

        cdf = max(ecdf) * norm.cdf(x, loc=y.mean(), scale=y.std())
        ax.plot(x, cdf, linestyle='dashed')
        ax.set_title(f"eCDF vs CDF of Stock Price Movements for {ticker}, {start_date} - {end_date}")

    fig.show()
    if image_path:
        if image_path.exists() and image_path.is_dir():
            image_path = image_path.joinpath(f"{ticker}_{start_date}-{end_date}_cdf.png")
        fig.savefig(image_path)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "input_df",
        type=Path
    )
    arg_parser.add_argument(
        "-c", "--columns",
        choices=['close', 'change'],
        nargs='+',
        default=['change']
    )
    arg_parser.add_argument(
        "-i", "--image-path",
        type=Path,
        default=None
    )
    args = arg_parser.parse_args()
    df = pd.read_pickle(args.input_df)
    plot_dist(df, *args.columns, image_path=args.image_path)


if __name__ == '__main__':
    main()
