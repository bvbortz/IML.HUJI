import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df["Temp"] > -50]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print(mean_square_error(y_true, y_pred))
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:/Users/bvbor/PycharmProjects/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]
    israel_df["Year"] = israel_df["Year"].astype(str)
    fig = px.scatter(israel_df, x='DayOfYear', y='Temp',
                     color='Year', title="average daily temperature in israel as a function of DayOfYear",
                     labels={"x": "number of day in the year", "y": "temperature"})
    fig.write_image("./graphs/israel_temp.png")
    israel_group = israel_df.groupby('Month')
    np_table = np.zeros((12, 2))
    np_table[:, 0] = np.array(np.arange(1, 13))
    for i in range(12):
        np_table[i, 1] = np.std(israel_group.get_group(i + 1)["Temp"].to_numpy())
    df_graph = pd.DataFrame(data=np_table, columns= ["Month", "std"])
    fig = px.bar(df_graph, x='Month', y='std', title="standard deviation in Israel per month",
                 labels={"Month": "month", "std": "standard deviation"})
    fig.write_image("./graphs/israel_temp_bar_std.png")
    # Question 3 - Exploring differences between countries

    country_groups = df.groupby("Country")
    df_graph = pd.DataFrame()
    for country in df["Country"].unique():
        month_group = country_groups.get_group(country).groupby('Month')
        np_table = np.zeros((12, 3))
        np_table[:, 0] = np.array(np.arange(1, 13))
        for i in range(12):
            np_table[i, 1] = np.std(month_group.get_group(i + 1)["Temp"].to_numpy())
            np_table[i, 2] = np.mean(month_group.get_group(i + 1)["Temp"].to_numpy())
        df_graph_local = pd.DataFrame(data=np_table, columns=["Month", "std", "mean"])
        df_graph_local["Country"] = country
        df_graph = pd.concat([df_graph, df_graph_local], axis=0)
    fig = px.line(df_graph, x="Month", y="mean", color="Country", title="Average temperature per month",
                  error_y="std",
                  labels={"Month": "month", "mean": "Average temperature"})
    fig.write_image("./graphs/avg_temp_month.png")
    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df["DayOfYear"], israel_df["Temp"])
    np_table = np.zeros((10, 2))
    np_table[:, 0] = np.array(np.arange(1, 11))
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(train_X.to_numpy(), train_y.to_numpy())
        np_table[k-1, 1] = np.round(poly.loss(test_X.to_numpy(), test_y.to_numpy()), decimals=2)
        print(np_table[k-1, 1])
    df_graph = pd.DataFrame(data=np_table, columns=["k", "loss"])
    fig = px.bar(df_graph, x='k', y='loss', title="loss of polynomial fitting for each k")
    fig.write_image("./graphs/poly_loss.png")


    # Question 5 - Evaluating fitted model on different countries
    poly = PolynomialFitting(5)
    poly.fit(israel_df["DayOfYear"], israel_df["Temp"])
    countries_loss = list()
    countries_list = list()
    for country in df["Country"].unique():
        if country != "Israel":
            country_df = country_groups.get_group(country)
            countries_loss.append(poly.loss(country_df["DayOfYear"], country_df["Temp"]))
            countries_list.append(country)
    table = {"Country": countries_list, "Loss": countries_loss}
    df_graph = pd.DataFrame(data=table)
    fig = px.bar(df_graph, x='Country', y='Loss', title="loss of Israel polynomial fitting for each country")
    fig.write_image("./graphs/country_poly_loss.png")

