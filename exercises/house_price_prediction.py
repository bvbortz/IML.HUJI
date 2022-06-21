from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

CURRENT_YEAR = 2015
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df = df.drop(["id", "long", "date", "lat"], axis=1)
    for label in ["yr_built", "yr_renovated", "zipcode"]:
        df[label] = df[label].astype(int)
    for label in ["yr_built", "price", "sqft_living", "sqft_lot", "floors", "sqft_living15", "sqft_lot15"]:
        df = df[df[label] > 0]
    for label in ["bedrooms", "bathrooms", "sqft_above", "sqft_basement", "yr_renovated"]:
        df = df[df[label] >= 0]
    for label in ["yr_built", "yr_renovated"]:
        df = df[df[label] <= CURRENT_YEAR]
    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 14))]
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df["recently_renovated"] = np.where(df["yr_renovated"] > CURRENT_YEAR - 10, 1, 0)
    df = df.drop("yr_renovated", axis=1)
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 800000]
    df = df[df["sqft_lot15"] < 500000]
    df = df[df["sqft_living"] < 10000]
    df = df[df["sqft_living15"] < 6000]
    return df.drop("price", axis=1), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    newX = X.loc[:, ~(X.columns.str.contains('zipcode_'))]
    for label in newX:
        rho = np.cov(newX[label], y)[0, 1] / (np.std(newX[label]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': newX[label], 'y': y}), x="x", y="y",
                         title="Response as a function of {} Pearson Correlation is {}".format(label,rho),
                         labels={"x": label + " Values", "y": "Response"})
        fig.write_image(output_path + "/{}.png".format(label))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:/Users/bvbor/PycharmProjects/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, "./graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fractions = np.linspace(0.1, 1, 91)
    lr = LinearRegression()
    confidence = np.zeros(91)
    mean_loss = np.zeros(91)
    for j in range(len(fractions)):
        loss = np.zeros(10)
        for i in range(10):
            train_X["price"] = train_y
            sampled_train = train_X.sample(frac=fractions[j])
            lr.fit(sampled_train.drop("price", axis=1), sampled_train.price)
            loss[i] = lr.loss(test_X.to_numpy(), test_y.to_numpy())
        confidence[j] = 2 * np.std(loss)
        mean_loss[j] = np.mean(loss)
    # frame = go.Frame(data=[go.Scatter(x=100 * fractions, y=mean_loss, mode="markers+lines", showlegend=False)],
    #                  layout=go.Layout(title="Linear regression mean loss over increasing fraction of the training set",
    #                                   xaxis={"title": "Fraction of training set"},
    #                                   yaxis={"title": "mean loss"}))
    # frame["data"] = (go.Scatter(x=100 * fractions, y=mean_loss+confidence, fill='tonexty', mode="lines",
    #                             line=dict(color="lightgrey"), showlegend=False),
    #                  go.Scatter(x=100 * fractions, y=mean_loss-confidence, fill=None, mode="lines",
    #                                  line=dict(color="lightgrey"), showlegend=False))
    # fig = go.Figure(data=frame["data"],
    #                 layout=go.Layout(
    #                     title=frame["layout"]["title"],
    #                     xaxis=frame["layout"]["xaxis"],
    #                     yaxis=frame["layout"]["yaxis"]))
    # fig = go.Figure(data=(go.Scatter(x=100 * fractions, y=mean_loss, mode="markers+lines", showlegend=False),
    #                       go.Scatter(x=100 * fractions, y=mean_loss+confidence, fill='tonexty', mode="lines",
    #                                  line=dict(color="lightgrey"), showlegend=False),
    #                       go.Scatter(x=100 * fractions, y=mean_loss-confidence, fill=None, mode="lines",
    #                                  line=dict(color="lightgrey"), showlegend=False)),
    #     layout=go.Layout(title="Linear regression mean loss over increasing fraction of the training set",
    #                        xaxis={"title": "Fraction of training set"},
    #                        yaxis={"title": "mean loss"}))
    fig = go.Figure([
        go.Scatter(
            name='Measurement',
            x=100 * fractions,
            y=mean_loss,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ),
        go.Scatter(
            name='Upper Bound',
            x=100 * fractions,
            y=mean_loss + confidence,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=100 * fractions,
            y=mean_loss - confidence,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='mean loss',
        title='Linear regression mean loss over increasing fraction of the training set',
        xaxis_title="Fraction of training set"
    )
    fig.write_image("./graphs/lr loss.png")


