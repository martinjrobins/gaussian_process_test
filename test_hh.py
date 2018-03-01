import pints
import pints.toy
import numpy as np
import pyDOE
import cma
import matplotlib.pylab as plt
from time import process_time
import gptest

# define the kernel and log likelihood


def fit_and_test_gp(N, dense=True):
    print('fit_and_test_gp(' + str(N) + ')')
    # create the model
    if False:
        model = pints.toy.HodgkinHuxleyIKModel()
        x0 = model.suggested_parameters()
        times = model.suggested_times()
    else:
        model = pints.toy.LogisticModel()
        x0 = [0.015, 500]
        times = np.linspace(0, 1000, 100)

    # sample the parameter space, treating time as another parameter
    lower = np.array([x / 10.0 for x in x0] + [times[0]], dtype='d')
    upper = np.array([x * 10.0 for x in x0] + [times[-1]], dtype='d')

    # training
    samples = pyDOE.lhs(model.dimension() + 1, samples=N)
    x = samples * (upper - lower) + lower

    # test
    samples = pyDOE.lhs(model.dimension() + 1, samples=1000)
    xt = samples * (upper - lower) + lower

    # call model simulate function with given samples
    print('starting model evaluations')
    y = np.apply_along_axis(lambda x: model.simulate(
        x[0:-1], [x[-1]]), 1, x).reshape(-1, 1)
    yt = np.apply_along_axis(lambda x: model.simulate(
        x[0:-1], [x[-1]]), 1, xt).reshape(-1, 1)
    print('finished model evaluations')

    print('x.shape =' + str(x.shape))
    print('y.shape =' + str(y.shape))

    # maximise log likelihood to get hyperparameters
    lower = np.append(lower, [0.0])
    upper = np.append(upper, [np.max(y)])

    print('starting hyperparameter optimisation')
    x0 = 0.5 * np.ones(len(upper))
    sigma = 0.9
    options = cma.CMAOptions()
    options.set('bounds', [[0 for x in lower], [1.0 for y in upper]])
    # options.set('transformation',[lambda x: x**2+1.2, None])

    if dense:

        def kernel(xi, xj, theta):
            scaled_theta = theta * (upper - lower) + lower
            invth2 = 1.0 / scaled_theta[0:-1]**2
            sqdist = np.sum(xi**2 * invth2, 1).reshape(-1, 1) + np.sum(
                xj**2 * invth2, 1) - 2 * np.dot(xi * invth2, xj.T)
            return scaled_theta[-1]**2 * np.exp(-sqdist / 2)
            return K

        def neg_marginal_log_likelihood(theta):
            K = kernel(x, x, theta) + 1e-5 * np.identity(x.shape[0])
            L = np.linalg.cholesky(K)
            logDetK = 2 * np.sum(np.log(np.diagonal(L)))
            invLy = np.linalg.solve(L, y)
            mll = 0.5 * np.dot(invLy.T, invLy) + 0.5 * logDetK + \
                x.shape[0] / 2 * np.log(2 * np.pi)
            return mll[0, 0]

        es = cma.CMAEvolutionStrategy(x0, sigma, options)
        t1 = process_time()
        es.optimize(neg_marginal_log_likelihood)
        t2 = process_time()
        es.result_pretty()
        theta = es.result.xbest * (upper - lower) + lower
        time_per_mll = (t2 - t1) / es.result.evaluations
        print('dense time_per_mll = ' + str(time_per_mll))

        K = kernel(x, x, es.result.xbest) + 1e-6 * np.identity(x.shape[0])

        # predict using test points
        Ks = kernel(x, xt, es.result.xbest)
        L = np.linalg.cholesky(K)
        Lk = np.linalg.solve(L, Ks)
        mu = np.dot(Lk.T, np.linalg.solve(L, y))
        error = np.sqrt(np.sum((mu - yt)**2) / np.sum(yt**2))
        print('dense N = ' + str(len(y)) + ' and error = ' + str(error))

        # points we're going to make predictions at.
        x0 = [0.015, 950]
        values = model.simulate(x0, times)
        xs = np.zeros((len(times), len(x0) + 1))
        for i, t in enumerate(times):
            xs[i, 0:len(x0)] = x0
            xs[i, -1] = t
        Ks = kernel(x, xs, es.result.xbest)
        Kss = kernel(xs, xs, es.result.xbest)

        # compute the mean at our test points.
        # L = np.linalg.cholesky(K)
        Lk = np.linalg.solve(L, Ks)
        mu = np.dot(Lk.T, np.linalg.solve(L, y)).reshape(-1)
        print(np.sqrt(np.sum((mu - values)**2) / len(values)))

        # compute the variance at our test points.
        s2 = np.diag(Kss) - np.sum(Lk**2, axis=0)
        s = np.sqrt(s2)

        plt.clf()
        plt.plot(times, values)
        plt.gca().fill_between(times, mu - 3 * s, mu + 3 * s, color="#dddddd")
        plt.plot(times, mu, 'r--', lw=2)
        plt.savefig('dense_fit_%d.pdf' % N)
    else:
        gpt = gptest.GaussianProcessTest(x)

        def neg_marginal_log_likelihood(theta):
            gpt.set_theta(theta)
            return gpt.calculate_mll()

        es = cma.CMAEvolutionStrategy(x0, sigma, options)
        t1 = process_time()
        es.optimize(neg_marginal_log_likelihood)
        t2 = process_time()
        es.result_pretty()
        theta = es.result.xbest * (upper - lower) + lower
        time_per_mll = (t2 - t1) / es.result.evaluations
        print('heirarchical time_per_mll = ' + str(time_per_mll))

        # predict using test points
        gpt.set_theta(es.result.xbest)
        mu = gpt.calculate_mu_at(xt)
        error = np.sqrt(np.sum((mu - yt)**2) / np.sum(yt**2))
        print('heirarchical N = ' + str(len(y)) + ' and error = ' + str(error))

        # points we're going to make predictions at.
        x0 = [0.015, 950]
        values = model.simulate(x0, times)
        xs = np.zeros((len(times), len(x0) + 1))
        for i, t in enumerate(times):
            xs[i, 0:len(x0)] = x0
            xs[i, -1] = t

        mu = gpt.calculate_mu_at(xs)
        print(np.sqrt(np.sum((mu - values)**2) / len(values)))
        s = gpt.calculate_s_at(xs)

        plt.clf()
        plt.plot(times, values)
        plt.gca().fill_between(times, mu - 3 * s, mu + 3 * s, color="#dddddd")
        plt.plot(times, mu, 'r--', lw=2)
        plt.savefig('dense_fit_%d.pdf' % N)

    return error


plt.figure()
Ns = np.logspace(np.log10(10), np.log10(500), 10)
error = [fit_and_test_gp(int(N)) for N in Ns]
plt.loglog(Ns, error)
plt.show()
