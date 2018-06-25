import matplotlib.pylab as plt
import numpy as np
from matplotlib import ticker

gaussian_matrix_solve_solve_time = np.loadtxt('gaussian_matrix_solve_solve_time.txt', skiprows=1)
gaussian_matrix_solve_setup_time = np.loadtxt('gaussian_matrix_solve_setup_time.txt', skiprows=1)

matern_matrix_solve_solve_time = np.loadtxt('matern_matrix_solve_solve_time.txt', skiprows=1)
matern_matrix_solve_setup_time = np.loadtxt('matern_matrix_solve_setup_time.txt', skiprows=1)

exp_matrix_solve_solve_time = np.loadtxt('exp_matrix_solve_solve_time.txt', skiprows=1)
exp_matrix_solve_setup_time = np.loadtxt('exp_matrix_solve_setup_time.txt', skiprows=1)

# tensor (N, sigma, dim, nbucket,precon)
N = np.array([1000, 2000, 4000, 8000, 16000])
sigma = np.array([0.1, 0.5, 0.9, 1.3, 1.7])
dim = np.array([1, 2, 3, 4, 5, 8, 10, 14])
nbucket = np.array([50, 150, 250, 350])
precon = ['no precon', 'swartz', 'nystrom']


def convert_to_tensor(data):
    print('converting to tensor')
    tensor = np.zeros((len(N), len(sigma), len(dim), len(nbucket), len(precon)))
    for row in data:
        iN = np.where(N == row[0])[0][0]
        isigma = np.where(sigma == row[1])[0][0]
        idim = np.where(dim == row[2])[0][0]
        inbucket = np.where(nbucket == row[4])[0][0]
        tensor[iN, isigma, idim, inbucket, 0] = row[6]
        tensor[iN, isigma, idim, inbucket, 1] = row[7]
        tensor[iN, isigma, idim, inbucket, 2] = row[8]
    return tensor


def convert_to_series(data):
    print('converting to series')
    series = np.zeros((3*len(data), 6))
    for i, row in enumerate(data):
        for j in range(3):
            series[3*i+j, :4] = row[[0, 1, 2, 4]]
            series[3*i+j, 4] = j
            series[3*i+j, 5] = row[6+j]
    return series


gaussian_matrix_setup_time_tensors = convert_to_tensor(gaussian_matrix_solve_setup_time)
gaussian_matrix_solve_time_tensors = convert_to_tensor(gaussian_matrix_solve_solve_time)
gaussian_matrix_setup_time_series = convert_to_series(gaussian_matrix_solve_setup_time)
gaussian_matrix_solve_time_series = convert_to_series(gaussian_matrix_solve_solve_time)
gaussian_matrix_solve_total_time = gaussian_matrix_setup_time_tensors + gaussian_matrix_solve_time_tensors

# total solve time versus N, sigma, dim, on for each nbucket
for i in range(len(nbucket)):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax8, ax10, ax14)) = plt.subplots(2, 4)
    for d, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax8, ax10, ax14]):
        ax.imshow(gaussian_matrix_solve_total_time[:, :, d, i, 0])
    fig.savefig('gaussian_matrix_solve_total_time_no_precon_%d.pdf' % nbucket[i])

    for d, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax8, ax10, ax14]):
        ax.imshow(gaussian_matrix_solve_total_time[:, :, d, i, 1])
    fig.savefig('gaussian_matrix_solve_total_time_swartz_%d.pdf' % nbucket[i])

    fig.clear()
    for d, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax8, ax10, ax14]):
        ax.imshow(gaussian_matrix_solve_total_time[:, :, d, i, 2])
    fig.savefig('gaussian_matrix_solve_total_time_nystrom_%d.pdf' % nbucket[i])
plt.close(fig)

# scatter plot x,y = time,N size=sigma color=dim, shape = precon


def my_scatter_radii(axes, data, indices, **kwargs):
    x_array = np.array(data[:, indices[0]], dtype=float)
    y_array = np.array(data[:, indices[1]], dtype=float)
    color_array = np.array(data[:, indices[2]], dtype=float)
    shape_array = np.array(data[:, indices[3]], dtype=float)
    radii_array = np.array(data[:, indices[4]], dtype=float)

    color_max = np.max(color_array)
    color_min = np.min(color_array)
    color_array = (color_array-color_min)/(color_max-color_min)
    shape_max = np.max(shape_array)
    shape_min = np.min(shape_array)
    shape_array = (3*(shape_array-shape_min)/(shape_max-shape_min)).astype(int)
    radii_max = np.max(radii_array)
    radii_min = np.min(radii_array)
    radii_array = (radii_array-radii_min)/(radii_max-radii_min)

    for (x, y, c, s, r) in zip(x_array, y_array, color_array, shape_array, radii_array):
        if s == 0:
            shape = pylab.Circle((x, y), radius=r, color=c, **kwargs)
        elif s == 1:
            shape = pylab.Rectangle((x-r, y-r), radius=r, color=c, **kwargs)
        else:
            shape = matplotlib.patches.CirclePolygon((x, y), radius=r, resolution=3, color=c, **kwargs)
        axes.add_patch(shape)
    return True


def my_parallel_coords(data, cols_names, colour_index, alpha_index):
    x = range(len(cols_names))
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    print('parallel coords')

    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15, 5))

    # Get min, max and range for each column
    # Normalize the data for each column
    data = np.copy(data)
    min_max_range = {}
    for col in x:
        mind, maxd, ranged = [data[:, col].min(), data[:, col].max(), np.ptp(data[:, col])]
        min_max_range[col] = [mind, maxd, ranged]
        data[:, col] = np.true_divide(data[:, col] - mind, ranged)

    # Plot each row
    for i, ax in enumerate(axes):
        print('\tplotting dim', i)
        for row in data:
            ax.plot(x, row, color=colours[int(round((len(colours)-1)*row[colour_index]))], alpha=0.5)
        ax.set_xlim([x[i], x[i+1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[dim]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = data[:, dim].min()
        norm_range = np.ptp(data[:, dim])
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols_names[dim]])

    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols_names[-2], cols_names[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    return fig


names = ['N', 'sigma', 'dim', 'nbucket', 'precon', 'time']
# for i in range(6):
#    fig = my_parallel_coords(gaussian_matrix_setup_time_series+gaussian_matrix_solve_time_series, ['N', 'sigma', 'dim', 'nbucket', 'precon', 'time'], i, 5)
#    fig.savefig('gaussian_matrix_solve_total_time_parallel_color_by_precon%d.pdf' % i)
# my_scatter_radii(plt.gca(), gaussian_matrix_setup_time_series+gaussian_matrix_solve_time_series, [0, 5, 2, 4, 1])
#    plt.close(fig)

# plot subplots time versus N for sigma, dim, each precon a diff color
for nb in range(len(nbucket)):
    print('plotting', nb)
    fig, axs = plt.subplots(len(sigma), len(dim), sharey=True, sharex=True)
    for i in range(len(sigma)):
        for j in range(len(dim)):
            row = len(sigma)-1-i
            for k in range(3):
                axs[row, j].loglog(N, gaussian_matrix_solve_total_time[:, i, j, nb, k].reshape(-1), label=precon[k])
            if row != len(sigma)-1:
                axs[row, j].set_xticks([])
            if j == 0:
                axs[row, j].set_ylabel('s=%f' % sigma[i])
            if j != 0:
                axs[row, j].set_yticks([])
            if row == len(sigma)-1:
                axs[row, j].set_xlabel('dim=%d' % dim[j])
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig('gaussian_matrix_solve_total_time_%d.pdf' % nbucket[nb])
    plt.close(fig)
