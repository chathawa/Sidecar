from matplotlib import pyplot as plt

from sidecar.stats.cdf import *


def plot_dist(df: pd.DataFrame, *columns, image_path=None):
    fig, axes = plt.subplots(1, len(columns))

    if len(columns) == 1:
        axes = (axes,)

    ticker = df['ticker'].values[0]
    start_date, end_date = min(df['date']), max(df['date'])

    for label, ax in zip(columns, axes):
        y = np.log(df[label].to_numpy())
        x = x_steps(y, 100)

        ecdf_y = ecdf(x, y)
        ax.plot(x, ecdf_y, color='red')

        ax.plot(x, cdf(x, y, scale=max(ecdf_y)), linestyle='dashed')
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
