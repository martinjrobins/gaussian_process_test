import scipy.optimize as opt
import numpy as np
import scipy.linalg as linalg
import matplotlib.pylab as plt
from matplotlib import ticker

N = 20
sigma = 0.1
x = np.linspace(-5, 5, N)
xi, xj = np.meshgrid(x, x)
true_theta = np.array([1.0, 1.0, 0.1])

# covariance matrix to generate data
trueC = true_theta[0]**2 * np.exp(
    -(xi - xj)**2 / (2.0 * true_theta[1]**2)) + true_theta[2]**2 * np.identity(N)

# generate data
# y = np.random.multivariate_normal(
#    x, trueC) + np.random.normal(scale=sigma, size=N)
y = np.random.multivariate_normal(np.zeros(N), trueC)
print('y = ', y)
# use GP regression


def neg_marginal_log_likelihood(theta):
    K = theta[0]**2 * np.exp(
        -(xi - xj)**2 / (2.0 * theta[1]**2)) + theta[2]**2 * np.identity(N)
    gradK = [np.zeros((N, N)) for i in range(len(theta))]
    gradK[0] = 2 * theta[0] * np.exp(-(xi - xj)**2 / (2.0 * theta[1]**2))
    gradK[1] = (xi - xj)**2 / (theta[1]**3) * theta[0]**2 * np.exp(
        -(xi - xj)**2 / (2.0 * theta[1]**2))
    gradK[2] = 2 * theta[2] * np.identity(N)
    L = linalg.cho_factor(K)
    logDetK = 2 * np.sum(np.log(np.diagonal(L[0])))
    invKy = linalg.cho_solve(L, y)
    mll = 0.5 * np.dot(y, invKy) + 0.5 * logDetK + N / 2 * np.log(2 * np.pi)
    gradMll = np.zeros(len(theta))
    for i in range(len(theta)):
        invKgradK = linalg.cho_solve(L, gradK[i])
        gradMll[i] = -0.5 * y.dot(
            linalg.cho_solve(L, gradK[i].dot(invKy))) + 0.5 * np.trace(invKgradK)
    return mll, gradMll

M = 40
ll = np.zeros((M, M))
v0 = np.zeros((M, M))
v1 = np.zeros((M, M))
v2 = np.zeros((M, M))
th0_points = np.logspace(
    np.log10(true_theta[0] / 10), np.log10(true_theta[0] * 10), M)
th1_points = np.logspace(
    np.log10(true_theta[1] / 10), np.log10(true_theta[1] * 10), M)
th2_points = np.logspace(
    np.log10(true_theta[2] / 10), np.log10(true_theta[2] * 10), M)

X, Y = np.meshgrid(th1_points, th2_points)
for i in range(M):
    for j in range(M):
        ll[i, j], [v0[i, j], v1[i, j], v2[i, j]] = neg_marginal_log_likelihood(
            [true_theta[0], X[i, j], Y[i, j]])


plt.contour(X, Y, ll, np.linspace(1, 40, 20))

plt.colorbar()
scale = np.sqrt(np.multiply(v1, v1) + np.multiply(v2, v2))
v1[scale > 100] = 0
v2[scale > 100] = 0
plt.quiver(X, Y, v1, v2)
plt.xscale("log")
plt.yscale("log")
plt.show()

x0 = np.array(true_theta * 10)
res = opt.minimize(neg_marginal_log_likelihood, x0,
                   jac=True, options={'disp': True})
print(res)
