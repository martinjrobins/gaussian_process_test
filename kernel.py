import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from math import sqrt

x = np.linspace(0, 1, 1000)


def matern(x, sigma):
    m_scale = sqrt(3.0)/sigma
    r = abs(x-0.5)
    return (1.0 + m_scale * r) * np.exp(-r * m_scale)


def gaussian(x, sigma):
    m_scale = 1.0/sigma**2
    r = abs(x-0.5)
    return np.exp(-r**2 * m_scale)


def exponential(x, sigma):
    m_scale = 1.0/sigma
    r = abs(x-0.5)
    return np.exp(-r * m_scale)


fig, ax = plt.subplots()
sigma_delta = 0.01
line_g, = ax.plot(x, gaussian(x, sigma_delta), label='gaussian')
line_m, = ax.plot(x, matern(x, sigma_delta), label='matern')
line_e, = ax.plot(x, exponential(x, sigma_delta), label='exponential')
ax.set_ylim(0, 1)
ax.legend()


def init():  # only required for blitting to give a clean slate.
    line_g.set_ydata([np.nan] * len(x))
    line_m.set_ydata([np.nan] * len(x))
    line_e.set_ydata([np.nan] * len(x))
    return (line_g, line_m, line_e)


def animate(i):
    line_g.set_ydata(gaussian(x, (i+1)*sigma_delta))  # update the data.
    line_m.set_ydata(matern(x, (i+1)*sigma_delta))  # update the data.
    line_e.set_ydata(exponential(x, (i+1)*sigma_delta))  # update the data.
    return (line_g, line_m, line_e)


ani = animation.FuncAnimation(
    fig, animate, frames=200, init_func=init, blit=True)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
