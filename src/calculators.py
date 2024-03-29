import time

from scipy.integrate import trapezoid
from src.drawers import *
from src.area import *
from scipy.optimize import curve_fit


def dispersion_func(k, vcos):
    return (np.sqrt(9.81 * k) - k * vcos) / 2 / np.pi


@njit
def bilinear_interpol(back, mask_div, mask_mod):

    """
    bilinear interpolation to cartesian directly from polar data
    :param back: input backscatter in polar
    :param mask_div: array of numbers of cells connecting polar and cartesian
    :param mask_mod: array of coefficients of cells connecting polar and cartesian

    using of njit is necessary!!!
    """
    res = np.zeros((mask_div.shape[1], mask_div.shape[2]))
    for i in range(mask_div.shape[1]):
        for j in range(mask_div.shape[2]):
            res[i, j] = (1 - mask_mod[0, i, j]) * (1 - mask_mod[1, i, j]) * \
                        back[mask_div[0, i, j], mask_div[1, i, j]] + \
                        (1 - mask_mod[0, i, j]) * mask_mod[1, i, j] * \
                        back[mask_div[0, i, j], mask_div[1, i, j] + 1] + \
                        mask_mod[0, i, j] * (1 - mask_mod[1, i, j]) * \
                        back[mask_div[0, i, j] + 1, mask_div[1, i, j]] + \
                        mask_mod[0, i, j] * mask_mod[1, i, j] * back[
                            mask_div[0, i, j] + 1, mask_div[1, i, j] + 1]
    return res


@njit
def closest_interpol(back: np.ndarray, mask_div: np.ndarray):
    """
    interpolation by closest element to cartesian directly from polar data
    :param back: input backscatter in polar
    :param mask_div: array of numbers of cells connecting polar and cartesian

    using of njit is necessary!!!
    """
    res = np.zeros((mask_div.shape[1], mask_div.shape[2]))
    for i in range(mask_div.shape[1]):
        for j in range(mask_div.shape[2]):
            res[i, j] = back[mask_div[0, i, j], mask_div[1, i, j]]
    return res


@njit
def make_circle(size: int):
    """
    making a circle mask for Radon transform to avoid artefacts
    :param size: size of research area (only square)
    """
    res = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if (x - size // 2) * (x - size // 2) + (y - size // 2) * (y - size // 2) < (size // 2) * (size // 2):
                res[x, y] = 1.
    return res


def _zero_array(array: np.ndarray, ind: int, window: int):
    """
    zeroing slice of array for [ind - window, ind + window]
    @param array: input 1D array
    @param ind: index of center of zeroes
    @param window: window for zeroing
    @return: copy of input array with zeroes on [ind - window, ind + window]
    """
    if ind - window < 0:
        array[ind - window:] = np.zeros(window - ind)
        array[0: ind] = np.zeros(ind)
    else:
        array[ind - window: ind] = np.zeros(window)
    if ind + window > array.shape[0]:
        array[ind:] = np.zeros(array.shape[0] - ind)
        array[0: ind + window - array.shape[0]] = np.zeros(ind + window - array.shape[0])
    else:
        array[ind: ind + window] = np.zeros(window)


def find_main_directions(radon_array: np.ndarray, num_peaks: int, window: int):
    """
    search num_peaks (or less) main wave direction
    :param radon_array: input 3d array result of Radon transform
    :param num_peaks: maximal number of directions in loop
    :param window: if AN is found direction, then in [AN - window, AN + window] new direction won't find
    """

    copy = np.mean(np.std(radon_array, axis=1), axis=0)

    grad = np.gradient(copy)
    ans = []  # array for obtained angles of peaks
    dirs = []  # array for obtained directions (forward / backward correlation coefficients) for angles
    for i in range(num_peaks):
        if i == 0:
            ans.append(np.argmax(copy))  # first peak
        else:
            an = np.argmax(copy)
            grad = np.diff(np.sign(grad[min(an, ans[-1]): max(an, ans[-1])]))
            if np.sum(np.abs(grad[grad != 0.])) != 0:  # check on local extremum
                ans.append(an)
            else:
                break

        _zero_array(copy, int(ans[i]), window)  # cut peak with \pm window for search next peak

        if i != 0:
            if np.abs(ans[i] - ans[i - 1]) % 360 < 2 * window:
                ans.pop()

    for i in range(len(ans)):
        dirs.append(calc_forward_toward(radon_array[:, :, ans[i]], 25))
        ans[i] *= -1
        ans[i] += 180
        if dirs[-1] > 0:
            ans[i] += 180
        if ans[i] > 360:
            ans[i] -= 360
        if ans[i] < 0:
            ans[i] += 360

    dirs = np.array(dirs)
    ans = np.array(ans)
    inds = np.abs(dirs).argsort()
    dirs = dirs[inds[::-1]]
    ans = ans[inds[::-1]]

    return ans, dirs


# def calc_autocorr(array: np.ndarray):
#    """
#    calculating a correlation coefficients for determination forward or backward
#    @param array: slice of radon signal by obtained angle
#    @return: relative correlation
#    """

#    l_arr = np.zeros(3)
#    r_arr = np.zeros(3)

#    for t in range(array.shape[0] - 1):
#        for i in range(l_arr.shape[0]):
#            r_arr[i] += np.corrcoef(array[t], np.roll(array[t + 1], 10 + 10 * i))[0, 1]  # roll forward
#            l_arr[i] += np.corrcoef(array[t], np.roll(array[t + 1], -10 - 10 * i))[0, 1]  # roll backward

#    # there is a comparing previous shot with next, next shot rolling forward and backward,
#    # by correlation determines direction of next shot relatively previous
#    res = np.max(r_arr - l_arr)

#    return res / (array.shape[0])


def calc_forward_toward(array: np.ndarray, win: int):
    toward = np.zeros(array.shape[0], dtype=int)
    copy = np.copy(array[0])
    peak_num = 5  # number of peaks
    c_win = 2  # coefficient in window for zeroing
    argmax_old = np.zeros(peak_num, dtype=int)

    for i in range(peak_num):
        argmax_old[i] = np.argmax(copy)
        _zero_array(copy, argmax_old[i], c_win * win)

    for t in range(1, array.shape[0]):

        copy = np.copy(array[t])
        argmax_new = np.zeros(peak_num, dtype=int)

        for i in range(peak_num):
            argmax_new[i] = np.argmax(copy[max(0, argmax_old[i] - win): min(argmax_old[i] + win, len(copy))]) - win

            if argmax_new[i] >= 0:
                toward[t] += 1
            else:
                toward[t] -= 1

        for i in range(peak_num):
            argmax_old[i] = np.argmax(copy)
            _zero_array(copy, argmax_old[i], c_win * win)

    return np.sum(toward) # >= 0


@njit
def calc_abs_wave_vector(array: np.ndarray) -> np.ndarray:
    """
    converting from (k_x, k_y, omega) to (|k|, omega)
    @param array: input data in (k_x, k_y, omega)
    """
    res = np.zeros(shape=(array.shape[0], array.shape[1]), dtype=float64)
    # loop for k_x and k_y
    for k_x in range(array.shape[1]):
        for k_y in range(array.shape[2]):
            k_abs = np.sqrt(k_x * k_x + k_y * k_y)
            k_abs_div = int(k_abs)
            k_abs_mod = k_abs - k_abs_div
            # analog of linear interpolation
            if k_abs_div < res.shape[1]:
                res[:, k_abs_div] += (1 - k_abs_mod) * array[:, k_x, k_y]
            if k_abs_div < res.shape[1] - 1:
                res[:, k_abs_div + 1] += k_abs_mod * array[:, k_x, k_y]
    return res


def calc_dispersion(name, array: np.ndarray, speed: float, df: int, turn_period: float, k_max: float):
    arr_2d = calc_abs_wave_vector(array)

    down = np.copy(arr_2d[:arr_2d.shape[0] // 2, :])
    up = np.copy(arr_2d[arr_2d.shape[0] // 2:, :])
    arr_2d[:arr_2d.shape[0] // 2, :] += np.flip(up, axis=0)
    arr_2d[arr_2d.shape[0] // 2:, :] += np.flip(down, axis=0)

    fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.3, wspace=.3)
    axs.get_xaxis().set_tick_params(which='both', direction='in')
    axs.get_yaxis().set_tick_params(which='both', direction='in')
    plt.rc('axes', axisbelow=True)
#
    axs.imshow(arr_2d, origin='lower', cmap='gnuplot', interpolation='None', aspect=0.6,
               extent=[0, k_max, 0, 1 / turn_period])
    axs.set_xlabel(r"Wave number $k$, rad/m")
    axs.set_ylabel(r"Frequency $f$, 1/s")

    max_freq = np.argmax(arr_2d, axis=0)

    grad = np.gradient(max_freq)

    if np.max(max_freq) > arr_2d.shape[0] // 2 - 10 and np.sum(grad[np.argmax(max_freq) + 1:] < 0):
        max_freq[np.argmax(max_freq) + 1:] *= -1
        max_freq[np.argmax(max_freq) + 1:] += arr_2d.shape[0]

    if np.mean(max_freq) < arr_2d.shape[0] // 10:
        print("ATTENTION ATTENTION ATTENTION ATTENTION")

    mask = np.abs(grad) < 3 * np.median(np.abs(grad))
    mask *= max_freq > arr_2d.shape[0] // 10
    mask *= max_freq < 9 * arr_2d.shape[0] // 10
    max_freq[0] = 0
    mask[0] = True

    popt, _ = curve_fit(dispersion_func, np.linspace(0, k_max, arr_2d.shape[1])[mask],
                        max_freq[mask] / arr_2d.shape[0] / turn_period,
                        sigma=1 / np.max(arr_2d, axis=0)[mask], absolute_sigma=True)
#
    axs.scatter(np.linspace(0, k_max, arr_2d.shape[1]), max_freq / arr_2d.shape[0] / turn_period, s=2)
    axs.scatter(np.linspace(0, k_max, arr_2d.shape[1])[mask], max_freq[mask] / arr_2d.shape[0] / turn_period, s=2)
    axs.plot(np.linspace(0, k_max, arr_2d.shape[1]), dispersion_func(np.linspace(0, k_max, arr_2d.shape[1]), 0),
             ls='--')
    axs.plot(np.linspace(0, k_max, arr_2d.shape[1]), dispersion_func(np.linspace(0, k_max, arr_2d.shape[1]), *popt),
              ls='--')

    l_mark = []
    r_mark = []
    l_mark2 = []
    r_mark2 = []

    noise = np.copy(arr_2d)

    for k in range(0, arr_2d.shape[1]):

        if k > 0:
            arr_2d[:, k] *= (k ** (-1.15))
            noise[:, k] *= (k ** (-1.15))

        freq = dispersion_func(k / arr_2d.shape[1] * k_max, *popt) * 256 * turn_period
        df_new = df
        # df_new = df * np.log(np.max(arr_2d[:, k])) / np.log(np.max(arr_2d))
        #if (freq > arr_2d.shape[0]) or (k > arr_2d.shape[1] - 10):
        #    df_new = 1
        #else:
        #    df_new = df * np.log(arr_2d[int(freq), k]) / np.log(np.max(arr_2d))
        ##
        #if (freq > 2 * df) and (freq < arr_2d.shape[0] - 2 * df):
        #    df_new = df * (np.log(arr_2d[round(freq), k]) - np.log(
        #        np.min(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, k]))) / (
        #                     np.log(np.max(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, :])) - np.log(
        #                 np.min(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, k])))

        left = max(round(freq - df_new), 0)
        l_mark.append(left)
        right = min(round(freq + df_new + 1), 256)
        r_mark.append(right)

        if right > left:
            noise[left: right, k] = np.zeros(right - left)

        freq = 255 - dispersion_func(k / arr_2d.shape[1] * k_max, *popt) * 256 * turn_period
        # df_new = df * np.log(np.max(arr_2d[:, k])) / np.log(np.max(arr_2d))
        #if (freq > arr_2d.shape[0]) or (k > arr_2d.shape[1] - 10):
        #    df_new = 1
        #else:
        #    df_new = df * np.log(arr_2d[int(freq), k]) / np.log(np.max(arr_2d))
        ##
        #if (freq > 2 * df) and (freq < arr_2d.shape[0] - 2 * df):
        #    df_new = df * (np.log(arr_2d[round(freq), k]) - np.log(
        #        np.min(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, k]))) / (
        #                     np.log(np.max(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, :])) - np.log(
        #                 np.min(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, k])))

        left = max(round(freq - df_new), 0)
        l_mark2.append(left)
        right = min(round(freq + df_new + 1), 256)
        r_mark2.append(right)

        if right > left:
            noise[left: right, k] = np.zeros(right - left)

    axs.plot(np.linspace(0, k_max, 32), np.array(l_mark) / 256 / turn_period, color='white', linewidth=1)
    axs.plot(np.linspace(0, k_max, 32), np.array(r_mark) / 256 / turn_period, color='white', linewidth=1)
    axs.plot(np.linspace(0, k_max, 32), np.array(l_mark2) / 256 / turn_period, color='white', linewidth=1)
    axs.plot(np.linspace(0, k_max, 32), np.array(r_mark2) / 256 / turn_period, color='white', linewidth=1)
    plt.savefig(PATH + "pics/" + name[-7:-3] + "0.png", bbox_inches='tight', dpi=700)
    plt.show()

    signal = arr_2d - noise
    # plt.imshow(signal, origin='lower', cmap='gnuplot', interpolation='None', aspect=0.6,
    #            extent=[0, k_max, 0, 1 / turn_period])
    # plt.show()
    # plt.imshow(arr_2d, origin='lower', cmap='gnuplot', interpolation='None', aspect=0.6,
    #            extent=[0, k_max, 0, 1 / turn_period])
    # plt.show()
    # plt.imshow(noise, origin='lower', cmap='gnuplot', interpolation='None', aspect=0.6,
    #            extent=[0, k_max, 0, 1 / turn_period])
    # plt.show()

    lenght = 2 * np.pi / (np.argmax(signal) % signal.shape[0] / signal.shape[0] * k_max)

    int_ind_k = 0
    int_ind_f = 0

    ss = trapezoid(signal[int_ind_k:])
    nn = trapezoid(noise[int_ind_k:])

    freq_specter = ss / nn
    m0 = trapezoid(ss[int_ind_f:]) / trapezoid(nn[int_ind_f:])
    m1 = trapezoid(ss[int_ind_f:] * np.linspace(0, 1 / turn_period, ss[int_ind_f:].shape[0])) / trapezoid(
        nn[int_ind_f:] * np.linspace(0, 1 / turn_period, ss[int_ind_f:].shape[0]))

    return m0, m1, freq_specter, np.arccos(max(min(popt[0] / speed, 1), -1)) / np.pi * 180, lenght
