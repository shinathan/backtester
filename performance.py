"""Most functions were taken from Section 13 from the notebook series. The fills_to_trade is the most difficult function. See _dissection.ipynb to understand it."""

# The shortcut to collapse all functions is CTRL+K, CTRL+0
# The shortcut to unfold is CTRL+K, CTRL+J

import numpy as np
from math import floor, ceil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
from times import get_market_dates
from IPython.display import display, HTML
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def calculate_annual_return(portfolio_log):
    # In percentage (base 100)
    cum_returns_gross = portfolio_log["return_cum"] * 0.01 + 1
    total_length = portfolio_log.index[-1] - portfolio_log.index[0]
    annual_return = (cum_returns_gross[-1]) ** (1 / (total_length.days / 365)) - 1
    return round(annual_return * 100, 1)


def calculate_sortina(portfolio_log, risk_free=0):
    # Riskfree also in base 100 percentage.
    returns = create_daily_portfolio_log(portfolio_log)["return"] * 0.01
    annual_mean = (calculate_annual_return(portfolio_log) - risk_free) * 0.01
    annual_downward_std = returns[returns < 0].std() * np.sqrt(252)
    return round(annual_mean / annual_downward_std, 2)


def calculate_sharpe(portfolio_log, risk_free=0):
    # Riskfree also in base 100 percentage.
    returns = create_daily_portfolio_log(portfolio_log)["return"] * 0.01
    annual_mean = (calculate_annual_return(portfolio_log) - risk_free) * 0.01
    annual_downward_std = returns.std() * np.sqrt(252)
    return round(annual_mean / annual_downward_std, 2)


def calculate_drawdowns(portfolio_log):
    """
    Get drawdown Series, maximum DD and maximum duration. Base 100 for percentages.

    Returns:
        (Series, float, datetime)
    """
    cum_returns_gross = portfolio_log["return_cum"] * 0.01 + 1
    maximum_gross_return = cum_returns_gross.cummax()
    drawdown = 1 - cum_returns_gross / maximum_gross_return

    ATH_series = drawdown[drawdown == 0]
    durations = (
        ATH_series.index[1:].to_pydatetime() - ATH_series.index[:-1].to_pydatetime()
    )
    drawdown = round(drawdown, 3)

    return drawdown * 100, drawdown.max() * 100, durations.max()


def fills_to_trades(fills_log):
    """Converts the fills log to a trade log. The explanation of this code is given in my other repo.

    Args:
        fills (DataFrame): the fills log

    Returns:
        DataFrame: the trade log
    """
    trade_log = pd.DataFrame(
        columns=[
            "datetime_in",
            "symbol",
            "side",
            "quantity",
            "entry",
            "exit",
            "datetime_out",
            "fees",
            "net P/L %",
            "net P/L $",
            "remaining_qty",
        ]
    )
    for dt, trade in fills_log.iterrows():
        symbol = trade["symbol"]
        side = trade["side"]
        opposite_side = "SELL" if side == "BUY" else "BUY"
        quantity = trade["quantity"]
        fill_price = trade["fill_price"]
        fees = trade["fees"]

        current_position_in_symbol_opposite = trade_log[
            (trade_log["symbol"] == symbol)
            & (trade_log["side"] == opposite_side)
            & (trade_log["remaining_qty"] > 0)
        ]
        if len(current_position_in_symbol_opposite) == 0:
            # If no open trades in this symbol in the opposite direction, create new trade
            trade_log.loc[len(trade_log)] = [
                dt,
                symbol,
                side,
                quantity,
                fill_price,
                np.nan,
                np.nan,
                fees,
                np.nan,
                np.nan,
                quantity,
            ]
        else:
            # Else we (partially) close the trade(s) and create a new trade if a net position remains. Using FIFO.
            for index, open_trade in current_position_in_symbol_opposite.iterrows():
                remaining_qty_open_trade = open_trade["remaining_qty"]
                already_filled_qty_open_trade = (
                    open_trade["quantity"] - open_trade["remaining_qty"]
                )
                current_average_fill = open_trade["exit"]

                # Partial close of open_trade
                if quantity < remaining_qty_open_trade:
                    if np.isnan(current_average_fill):
                        trade_log.loc[index, "exit"] = fill_price
                    else:
                        average_fill_exit = (
                            (current_average_fill * already_filled_qty_open_trade)
                            + (fill_price * quantity)
                        ) / (
                            already_filled_qty_open_trade + quantity
                        )  # Calculate new average fill
                        trade_log.loc[index, "exit"] = average_fill_exit

                    trade_log.loc[index, "remaining_qty"] -= quantity
                    trade_log.loc[index, "fees"] += fees
                    break  # We don't have to look at the next trade

                # Full close of open_trade
                elif quantity >= remaining_qty_open_trade:
                    if np.isnan(current_average_fill):
                        trade_log.loc[index, "exit"] = fill_price
                    else:
                        average_fill_exit = (
                            (current_average_fill * already_filled_qty_open_trade)
                            + (fill_price * quantity)
                        ) / (
                            already_filled_qty_open_trade + quantity
                        )  # Calculate new average fill
                        trade_log.loc[index, "exit"] = average_fill_exit

                    trade_log.loc[index, "remaining_qty"] = 0
                    trade_log.loc[index, "fees"] += fees
                    trade_log.loc[index, "datetime_out"] = dt

                    if quantity == remaining_qty_open_trade:
                        break  # We don't have to look at the next trade
                    else:
                        quantity = (
                            quantity - remaining_qty_open_trade
                        )  # Calculate remaining quantity

                        # If we are at the end and there is still a remaining quantity, that is a new position
                        if index == len(current_position_in_symbol_opposite) - 1:
                            trade_log.loc[len(trade_log)] = [
                                dt,
                                symbol,
                                side,
                                quantity,
                                fill_price,
                                np.nan,
                                np.nan,
                                fees,
                                np.nan,
                                np.nan,
                                quantity,
                            ]
        trade_log["datetime_in"] = pd.to_datetime(trade_log["datetime_in"])
        trade_log["datetime_out"] = pd.to_datetime(trade_log["datetime_out"])
        trade_log["fees"] = round(trade_log["fees"], 2)
    return calculate_PNL_trade_log(trade_log)


def fills_to_trades_no_partials(fills_log):
    trade_log = pd.DataFrame(
        columns=[
            "datetime_in",
            "symbol",
            "side",
            "quantity",
            "entry",
            "exit",
            "datetime_out",
            "fees",
            "net P/L %",
            "net P/L $",
            "remaining_qty",
        ],
    )

    for dt, trade in fills_log.iterrows():
        symbol = trade["symbol"]
        side = trade["side"]
        opposite_side = "SELL" if side == "BUY" else "BUY"
        quantity = trade["quantity"]
        fill_price = trade["fill_price"]
        fees = trade["fees"]

        current_position_in_symbol_opposite = trade_log[
            (trade_log["symbol"] == symbol)
            & (trade_log["side"] == opposite_side)
            & (trade_log["remaining_qty"] > 0)
        ]

        if current_position_in_symbol_opposite.empty:
            # If no open trades in this symbol in the opposite direction, create new trade
            trade_log.loc[len(trade_log)] = [
                dt,
                symbol,
                side,
                quantity,
                fill_price,
                np.nan,
                np.nan,
                fees,
                np.nan,
                np.nan,
                quantity,
            ]
        else:
            for index, open_trade in current_position_in_symbol_opposite.iterrows():
                trade_log.at[index, "exit"] = fill_price
                trade_log.at[index, "remaining_qty"] -= quantity
                trade_log.at[index, "fees"] += fees
                trade_log.at[index, "datetime_out"] = dt

        trade_log["datetime_in"] = pd.to_datetime(trade_log["datetime_in"])
        trade_log["datetime_out"] = pd.to_datetime(trade_log["datetime_out"])
        trade_log["fees"] = round(trade_log["fees"], 2)
    return calculate_PNL_trade_log(trade_log)


def calculate_PNL_trade_log(trade_log):
    """Calculate the PNL for the trade log. Percentages in base 100.
    Args:
        trade_log (DataFrame): the trade log

    Returns:
        DataFrame: the trade log with PNL
    """
    trade_log["direction"] = np.where(trade_log["side"] == "BUY", 1, -1)
    trade_log["net P/L %"] = (
        100
        * (
            (
                (trade_log["quantity"] - trade_log["remaining_qty"])
                * trade_log["direction"]
                * (trade_log["exit"] - trade_log["entry"])
            )
            - trade_log["fees"]
        )
        / (trade_log["entry"] * trade_log["quantity"])
    )
    trade_log["net P/L $"] = (
        trade_log["net P/L %"] * 0.01 * (trade_log["entry"] * trade_log["quantity"])
    )

    trade_log["net P/L %"] = round(trade_log["net P/L %"], 2)
    trade_log["net P/L $"] = round(trade_log["net P/L $"], 2)
    return trade_log.drop(columns=["direction"])


def calculate_time_in_market(portfolio_log):
    """Calculates the exposure based on the portfolio_log values. So if you
    buy-and-hold 50% of SPY, then the exposure is 50%. However if you always
    close your positions before the logging moment, the exposure is 0%.

    Args:
        portfolio_log (DataFrame): the portfolio log

    Returns:
        float: the percentage of days that the strategy is in the market. Base 100.
    """
    portfolio_log = create_daily_portfolio_log(portfolio_log)
    return round(
        (portfolio_log["positions_value"] / portfolio_log["equity"]).mean() * 100, 0
    )


def calculate_average_trade_duration(trade_log):
    """Calculates the average time in a trade

    Args:
        trade_log (DataFrame): the trade log

    Returns:
        list(float, float, float): a list of the days, hours and minutes
    """
    tdelta = (trade_log["datetime_out"] - trade_log["datetime_in"]).sum() / len(
        trade_log
    )
    # days = tdelta.days
    # hours = tdelta.seconds // 3600
    # minutes = (tdelta.seconds // 60) % 60
    return tdelta


def calculate_fees_drag(portfolio_log, fills_log):
    """Calculates the fees drag per year. Percentage in base 100.

    Args:
        portfolio_log (DataFrame): the portfolio log
        fills_log (DataFrame): the fills log

    Returns:
        float: the fees drag per year
    """
    portfolio_log_daily = create_daily_portfolio_log(portfolio_log)

    fills_log = fills_log.copy()
    fills_log.index = fills_log.index.date
    fills_log_daily = fills_log.groupby(fills_log.index).aggregate({"fees": "sum"})

    portfolio_log_daily = portfolio_log_daily.merge(
        fills_log_daily, left_index=True, right_index=True, how="left"
    )
    portfolio_log_daily["fees"] = portfolio_log_daily["fees"].fillna(0)

    fees_per_day = (portfolio_log_daily["fees"] / portfolio_log_daily["equity"]).mean()
    return round(fees_per_day * 250 * 100, 1)


def calculate_average_profit(trade_log):
    """Calculates average profit per trade. Percentages in base 100.

    Args:
        trade_log (DataFrame): the trade log

    Returns:
        float: the value
    """
    return round(trade_log["net P/L %"].mean(), 2)


def calculate_trades_per_month(portfolio_log, trade_log):
    """Calculates average amount of trades per month

    Args:
        portfolio_log (DataFrame): the portfolio log
        trade_log (DataFrame): the trade log

    Returns:
        float: the value
    """
    # There are on average 21 trading days per month.
    total_months = (portfolio_log.index[-1] - portfolio_log.index[0]).days / 21
    return round(len(trade_log) / total_months, 1)


def calculate_profit_factor(trade_log):
    """Calculates the profit factor (avg win/avg loss)

    Args:
        trade_log (DataFrame): the trade log

    Returns:
        float: the profit factor
    """
    average_profit = trade_log[trade_log["net P/L %"] > 0]["net P/L %"].mean()
    average_loss = abs(trade_log[trade_log["net P/L %"] < 0]["net P/L %"].mean())
    return round(average_profit / average_loss, 2)


def calculate_winning_months(portfolio_log):
    """Calculate percentage of winning months. Percentage in base 100.

    Args:
        portfolio_log (DataFrame): the portfolio log

    Returns:
        float: the percentage
    """
    total_months = (portfolio_log.index[-1] - portfolio_log.index[0]).days / 30
    monthly_return = portfolio_log["return"].resample("1M").sum()
    return round(
        100 * monthly_return[monthly_return > 0].count() / len(monthly_return), 0
    )


def calculate_returns_per_year(portfolio_log):
    create_daily_portfolio_log(portfolio_log)["equity"]
    equity = portfolio_log[["equity"]].copy()
    equity["equity_start"] = equity["equity"]
    equity["equity_end"] = equity["equity"]

    equity = equity.resample("Y").agg({"equity_start": "first", "equity_end": "last"})
    equity["return"] = round((equity.equity_end / equity.equity_start - 1) * 100, 1)
    equity["return"] = equity["return"].astype(str) + "%"

    equity.index = equity.index.year
    equity.index.name = None
    equity = equity[["return"]]
    equity.columns = ["Return"]
    return equity


def display_stats(portfolio_log, fills_log, trade_log):
    avg_trade_duration = calculate_average_trade_duration(trade_log)
    days = avg_trade_duration.days
    hours = avg_trade_duration.seconds // 3600
    minutes = (avg_trade_duration.seconds // 60) % 60
    if days > 0:
        display_duration = f"{days}d, {round(avg_trade_duration.seconds/3600, 1)}h"
    else:
        display_duration = f"{hours}h{minutes}m"

    statistics = {
        "Annual return": f"{calculate_annual_return(portfolio_log)}%",
        "Sharpe": str(calculate_sharpe(portfolio_log)),
        "Sortina": str(calculate_sortina(portfolio_log)),
        "Winning months": f"{calculate_winning_months(portfolio_log)}%",
        "Time in market": f"{calculate_time_in_market(portfolio_log)}%",
        "Average profit": f"{calculate_average_profit(trade_log)}%",
        "Average duration per trade": f"{display_duration}",
        "Profit factor": str(calculate_profit_factor(trade_log)),
        "Trades/month": str(calculate_trades_per_month(portfolio_log, trade_log)),
        "Annual fees": f"{calculate_fees_drag(portfolio_log, fills_log)}%",
    }
    df = pd.DataFrame.from_dict(statistics, orient="index")
    df.columns = ["Stats"]
    return df


def display_side_by_side(dfs: list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    for df in dfs:
        output += df.style.set_table_attributes("style='display:inline'")._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


def display_stats_and_returns(portfolio_log, fills_log, trade_log):
    """Displays that strategy stats and annual returns per year."""
    statistics = display_stats(portfolio_log, fills_log, trade_log)
    annual_returns = calculate_returns_per_year(portfolio_log)
    if len(annual_returns) <= 10:
        display_side_by_side([statistics, annual_returns])
    else:
        display_side_by_side(
            [
                statistics,
                annual_returns.head(floor(len(annual_returns) / 2)),
                annual_returns.tail(ceil(len(annual_returns) / 2)),
            ]
        )


def create_daily_portfolio_log(portfolio_log):
    """
    Reindex and forward fill the portfolio log to the trading dates. This is to ensure
    no gaps if the portfolio log isn't logged every day.
    """
    portfolio_log = portfolio_log.copy()
    portfolio_log.index = portfolio_log.index.date

    market_dates = get_market_dates()
    market_dates_portfolio = list(
        filter(
            lambda day: (day >= portfolio_log.index[0])
            & (day <= portfolio_log.index[-1]),
            market_dates,
        )
    )

    portfolio_log = portfolio_log.reindex(market_dates_portfolio)
    portfolio_log[["equity", "cash", "return_cum"]] = portfolio_log[
        ["equity", "cash", "return_cum"]
    ].ffill()
    portfolio_log["positions_value"] = portfolio_log["positions_value"].fillna(0)
    portfolio_log["return"] = portfolio_log["return"].fillna(0)
    portfolio_log["positions"] = portfolio_log["positions"].fillna("{}")

    return portfolio_log


def plot_fig(portfolio_log, title):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(pad=1.5)

    # Returns
    ax1.plot(
        portfolio_log.index,
        portfolio_log["return_cum"],
        color="midnightblue",
        linewidth=1,
    )
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_title("Total return", fontsize=10, fontweight="bold")

    # Drawdown
    drawdowns, _, _ = calculate_drawdowns(portfolio_log)
    ax2.plot(drawdowns.index, -drawdowns, color="firebrick", linewidth=1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_title("Drawdown", fontsize=10, fontweight="bold")

    # Monthly returns
    monthly_return = portfolio_log["return"].resample("1M").sum()
    colors = ["firebrick" if ret < 0 else "g" for ret in monthly_return]
    monthly_return_index = monthly_return.index.values
    monthly_return_index[0] = portfolio_log.index[0]  # To make the x-axis align
    monthly_return_index[-1] = portfolio_log.index[-1]  # To make the x-axis align

    ax3.bar(monthly_return.index, monthly_return.values, width=15, color=colors)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax3.set_title("Monthly returns", fontsize=10, fontweight="bold")

    plt.show()


def plot_quintiles(
    trade_log, column, bins=5, title=None, aggregate_dict={"net P/L %": "mean"}
):
    quintiles = trade_log.groupby(pd.qcut(trade_log[column], bins)).aggregate(
        aggregate_dict
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(range(1, bins + 1), quintiles[next(iter(aggregate_dict))], color="green")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Quantile (low to high)")
    ax.set(xticks=range(1, bins + 1))
    ax.set_ylabel("Avg. net profit/trade")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    plt.show()


def plot_quantiles_periods(
    trade_log,
    column,
    bins=5,
    title=None,
    aggregate_dict={"net P/L %": "mean"},
    figsize=(8, 5),
):
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for ax, trade_log_split in zip(
        axes.reshape(-1), np.array_split(trade_log, axes.size)
    ):
        quintiles = trade_log_split.groupby(
            pd.qcut(trade_log_split[column], bins)
        ).aggregate(aggregate_dict)

        ax.bar(range(1, bins + 1), quintiles[next(iter(aggregate_dict))], color="green")
        start_date = trade_log_split["datetime_in"].iloc[0].strftime("%b %Y")
        ax.set_title(f">{start_date}")
        ax.set(xticks=range(1, bins + 1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()
