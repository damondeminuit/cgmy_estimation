import pandas as pd
import numpy as np
import os

DATA_FOLDER = "../data"


def load_returns(path, start_date=None, end_date=None):
    df = pd.read_csv(f"{DATA_FOLDER}/{path}")
    df["Date"] = pd.to_datetime(df["Date"])
    df.index = df["Date"]
    col = "Close" if "Last Price" not in df.columns else "Last Price"
    df = df.filter([col])
    df = df.sort_index()
    if start_date is not None and end_date is not None:
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    rets = np.log(df[col]).diff().dropna()
    return rets


def load_orderbook_trades(freq):
    """
    Load the trades from the orderbook dataset
    freq : str, the frequency of the data
    """
    path_to_orderbook = f"{DATA_FOLDER}/orderbook"
    files = os.listdir(path_to_orderbook)
    L = []

    # Load the prices for each day
    for file in files:
        if "message" not in file:
            continue

        updates = pd.read_csv(
            f"{path_to_orderbook}/{file}",
            names=["time", "type", "order_id", "volume", "price", "direction", "null"],
            low_memory=False,
        )
        trades = updates[updates["type"] == 4]  # select trades
        trades = trades.reset_index().drop(columns="index")
        base_date = pd.to_datetime(
            file.split("_")[1]
        )  # find the date in the name of the file
        trades["time"] = base_date + pd.to_timedelta(
            trades["time"], unit="s"
        )  # retrieve the correct timstamp
        trades = trades.filter(["time", "price"])
        trades.index = trades["time"]
        trades = trades.sort_index()
        trades_freq = trades.groupby(
            pd.Grouper(key="time", freq=freq)  # group by frequency
        ).last()  # select last trade for each frequency

        # create a full range of seconds for the trading day (assuming 9:30–16:00)
        market_open = base_date + pd.Timedelta(hours=9, minutes=30)
        market_close = base_date + pd.Timedelta(hours=16)
        all_seconds = pd.date_range(start=market_open, end=market_close, freq=freq)

        # reindex trades onto full timeline
        trades_full = (
            trades_freq.reindex(all_seconds)  # insert all missing seconds
            .rename_axis("time")
            .reset_index()
        )

        # handle missing prices: leave NaN, or forward-fill if appropriate
        trades_full["price"] = trades_full["price"].ffill()
        L.append(trades_full)

    trades = pd.concat(L)  # concatenate all the dates
    trades.index = trades["time"]
    trades = trades.sort_index()
    trades["rets"] = np.log(trades["price"]).diff()

    return trades["rets"].dropna()
