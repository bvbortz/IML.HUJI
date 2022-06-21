from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    np.random.seed(0)
    samples = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian()

    ug.fit(samples)
    print("({}, {})".format(ug.mu_, ug.var_))

    # Question 2 - Empirically showing sample mean is consistent
    distance = np.zeros(100)
    i = 1
    while 10*i <= 1000:
        ug.fit(samples[:10*i])
        distance[i-1] = (ug.mu_ - 10) ** 2
        i += 1
    num_of_samples = 10 * np.array(np.arange(1, 101))
    fig1 = px.line(x=num_of_samples, y=distance,
                  labels={'x': 'number of samples',
                          'y': 'squared error'},
                  title="error as a function of the number of samples")
    plotly.offline.plot(fig1)

    # Question 3 - Plotting Empirical PDF of fitted model
    x_vals = np.array(np.arange(-5, 20, 0.01))
    pdf_res = ug.pdf(x_vals)

    fig = px.line(x=x_vals, y=pdf_res,
                  labels={'x': 'x',
                          'y': 'f(x)'},
                  title="pdf of the estimated distribution")
    plotly.offline.plot(fig)



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    np.random.seed(0)
    mean1 = np.array([0, 0, 4, 0])
    cov1 = np.array([[1, 0.2, 0, 0.5],
                     [0.2, 2, 0, 0],
                     [0, 0, 1, 0],
                     [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean1, cov1, 1000)
    mg = MultivariateGaussian()
    mg.fit(samples)
    print("mean {} \n cov {}".format(mg.mu_, mg.cov_))

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    lls = np.zeros((f1.size, f3.size))
    i=0
    for mu1 in f1:
        j = 0
        for mu3 in f3:
            mean2 = np.array([mu1, 0, mu3, 0])
            lls[i, j] = mg.log_likelihood(mean2, cov1, samples)
            j += 1
        i += 1
    fig = go.Figure(go.Heatmap(x=f3, y=f1, z=lls),
              layout=go.Layout(title="log likelihood of different mu",
                               xaxis=dict(title="f3"),
                               yaxis=dict(title="f1")))
    plotly.offline.plot(fig)

    # Question 6 - Maximum likelihood
    num_index = np.argmax(lls.flatten())
    mu1_hat = f1[num_index // len(f1)]
    mu3_hat = f3[num_index % len(f3)]
    print("{}, {}".format(np.around(mu1_hat, decimals=3),
                          np.around(mu3_hat, decimals=3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
