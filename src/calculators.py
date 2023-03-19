from skimage.transform import radon
from scipy.integrate import trapezoid
import scipy.fft as sf
from src.drawers import *
from src.structures import *
from scipy.optimize import curve_fit


def dispersion_func(k, vcos):
    return (np.sqrt(9.81 * k) - k * vcos) / 2 / np.pi


@njit
def speed_bilinear_interpol(back, mask_div, mask_mod):
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
def speed_no_interpol(back: np.ndarray, mask_div: np.ndarray):
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
    :param one: flag for maximum two starts this function
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

    print(ans)

    for i in range(len(ans)):
        dirs.append(calc_autocorr(radon_array[:, :, ans[i]]))
        ans[i] *= -1
        ans[i] += 180
        if dirs[-1] < 0:
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


def calc_autocorr(array: np.ndarray):
    """
    calculating a correlation coefficients for determination forward or backward
    @param array: slice of radon signal by obtained angle
    @return: relative correlation
    """

    l = np.zeros(3)
    r = np.zeros(3)

    for t in range(array.shape[0] - 1):
        for i in range(l.shape[0]):
            r[i] += np.corrcoef(array[t], np.roll(array[t + 1], 10 + 10 * i))[0, 1]  # roll forward
            l[i] += np.corrcoef(array[t], np.roll(array[t + 1], -10 - 10 * i))[0, 1]  # roll backward

    # there is a comparing previous shot with next, next shot rolling forward and backward,
    # by correlation determines direction of next shot relatively previous
    res = np.max(r - l)

    return res / (array.shape[0])


def radon_process(array: np.ndarray, threshold: float, mask_circle: np.ndarray):
    """
    preprocess filtering (only threshold) and transforming by Radon
    @param array: input data
    @param threshold: if element more than 2 * threshold * mean, it becomes 1, else it 0
    @param mask_circle: a special mask that cuts a circle from a square for Radon transformation
    @return transformed data
    """
    radon_array = np.ndarray(shape=(array.shape[0], array.shape[1], 180), dtype=float)
    for t in range(array.shape[0]):
        # filtering
        # array[t][array[t] < 2 * threshold * np.mean(array[t])] = 0
        # array[t][array[t] != 0] = 1
        # radon transform
        radon_array[t] = radon(mask_circle * array[t])
    return radon_array


def calc_std(data_nc, start_ind: int, max_time: int, version: bool):
    """
    calculating dispersion of some serial shots to find more clean zone of data
    @param data_nc: input netcdf data
    @param start_ind: zero buoy time shot
    @param max_time: full number of recorded shots
    @param version: for old versions of variables names
    @return azimuth and distance of clear zone
    """

    bck_cntr = np.ndarray(shape=(4, 4096, 100), dtype=float)
    rho_bck_0 = 700
    for t in range(bck_cntr.shape[0]):
        if version:

            if t < max_time - 1:
                bck_cntr[t] = data_nc.variables["bsktr_radar"][t + start_ind][:,
                              rho_bck_0:rho_bck_0 + bck_cntr.shape[2]]
            else:
                bck_cntr[t] = data_nc.variables["bsktr_radar"][- t - start_ind + 2 * max_time - 1][:,
                              rho_bck_0:rho_bck_0 + bck_cntr.shape[2]]
        else:
            bck_cntr[t] = data_nc.variables["bsktr"][t + start_ind][:, rho_bck_0:rho_bck_0 + bck_cntr.shape[2]]
            if t < max_time:
                bck_cntr[t] = data_nc.variables["bsktr"][t + start_ind][:, rho_bck_0:rho_bck_0 + bck_cntr.shape[2]]
            else:
                bck_cntr[t] = data_nc.variables["bsktr"][- t - start_ind + 2 * max_time - 1][:,
                              rho_bck_0:rho_bck_0 + bck_cntr.shape[2]]

    std_array = np.std(bck_cntr, axis=0)
    # plt.imshow(std_array, cmap='Greys', origin='lower')
    # plt.show()

    std_array_smooth = np.zeros((int(bck_cntr.shape[1] / 64 - 1), int(bck_cntr.shape[2] / 10 - 1)))
    for i in range(std_array_smooth.shape[0] - 1):
        for j in range(std_array_smooth.shape[1] - 1):
            std_array_smooth[i, j] = np.mean(std_array[i * 64: i * 64 + 64, j * 10: j * 10 + 10])

    argmax = np.argmax(std_array_smooth)
    theta = (argmax // std_array_smooth.shape[1]) / std_array_smooth.shape[0] * 360
    rho = (argmax % std_array_smooth.shape[1] + rho_bck_0) * 1.875

    return theta, rho


def calc_back(data_nc, calc_time: int, start_ind: int, max_time: int, resolution: int, area_size: int, size_array: int,
              flag_area: int, an: float, version: bool):
    """
    calculating a cartesian data inside current area
    @param data_nc: input netcdf data
    @param calc_time: number of calculating turns
    @param start_ind: zero buoy time shot
    @param max_time: number of recorded turns
    @param resolution: resolution in converting from polar to cartesian
    @param area_size: size of research square in meters
    @param size_array: size of research square in pixels
    @param flag_area: 0 for Radon, 1 for reRadon, 2 for Fourier
    @param an: obtained by radon angle
    @param version: for old versions of variables names
    @return cartesian data inside area
    """
    if flag_area != 2:
        back = np.ndarray(shape=(calc_time, size_array, size_array), dtype=float)
    else:
        back = np.ndarray(shape=(calc_time, size_array, size_array), dtype=complex)

    ang_std = []

    for t in range(calc_time):

        if t % 60 == 0:
            angle_std, rad_std = calc_std(data_nc, start_ind, max_time, version)
            ang_std.append(angle_std)

            if np.abs(angle_std - 6) < 6:  # 6 degrees is gluing place, we need avoid it
                angle_std += np.sign(angle_std - 6) * 2 * np.abs(angle_std - 6)

        if flag_area == 0:
            zone = Area(area_size, area_size, rad_std, angle_std, 0)
        if flag_area == 1:
            zone = Area(area_size, area_size, 1360, 270, 0)
        if flag_area == 2:
            zone = Area(area_size, area_size, rad_std, angle_std, (-angle_std - an) % 360)
            # zone = Area(area_size, area_size, rad_std, angle_std, (-angle_std - an) % 360)

        if version:
            area_mask_div, area_mask_mod, min_max = make_area_mask(zone, data_nc.variables["rad_radar"][-1],
                                                                   data_nc.variables["rad_radar"].shape[0],
                                                                   data_nc.variables["theta_radar"].shape[0],
                                                                   resolution)
            if t < max_time:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][t + start_ind])
            else:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t - start_ind + 2 * max_time - 1])
        else:
            area_mask_div, area_mask_mod, min_max = make_area_mask(zone, data_nc.variables["radar_rad"][-1],
                                                                   data_nc.variables["radar_rad"].shape[0],
                                                                   data_nc.variables["radar_theta"].shape[0],
                                                                   resolution)
            if t < max_time:
                back_polar = np.transpose(data_nc.variables["bsktr"][t + start_ind])
            else:
                back_polar = np.transpose(data_nc.variables["bsktr"][- t - start_ind + 2 * max_time - 1])

        if flag_area != 2:
            back[t] = speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod)
        else:
            back[t] = sf.fft2(speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod))[:size_array,
                      :size_array]

    res_an = int(np.median(np.array(ang_std)))
    if res_an > 360:
        res_an -= 360
    if res_an < 0:
        res_an += 360

    return back, res_an


def cut_one_harm_3d(data: np.ndarray, df: int, turn_period: float, k_min: float, k_max: float, speed: float,
                    angle: float,
                    inverse=False, flag_speed=1):
    """
    cutting main harmonics \omega = \sqrt{gk} + kV
    @param data: input data after fourier transform with background and signal
    @param df: window of frequency for cutting
    @param turn_period: period of one radar's turn
    @param k_min: minimum of wave number
    @param k_max: maximum of wave number
    @param speed: speed of vessel in meters/sec
    @param angle: angle between speed and wave vector
    @param inverse: True if needed to mirror input array
    @param flag_speed: forward/backward flag
    return input array with zeroes on elements in dispersion relation
    """
    res = np.copy(data)
    for k_x in range(data.shape[1]):
        for k_y in range(data.shape[2]):
            alpha = np.arctan2(k_y, k_x) + np.pi * angle / 180  # + np.pi / 4
            # convert from wave vector in pix to wave number in rad/meters
            k_abs = np.sqrt(k_x ** 2 + k_y ** 2) / data.shape[1] * k_max + k_min
            # calculating a frequency by dispersion relation
            freq = np.sqrt(9.81 * k_abs) + flag_speed * k_abs * speed * np.cos(alpha)

            freq /= (2 * np.pi)
            # convert in index
            freq *= (turn_period * 256)
            freq = int(freq)

            if inverse:
                freq -= df

            left = max(freq - df, 0)
            right = min(freq + df + 1, 256)

            for f in range(left, right):
                if f > 127 + 0.25 * df or f < 127 - 0.25 * df:
                    dist_f = np.abs(f - freq - df // 2)
                    # special coefficients for smooth cutting
                    if dist_f <= df / 3:
                        res[f, k_x, k_y] = 0
                    elif dist_f <= 7 * df / 9:
                        res[f, k_x, k_y] *= 0.3
                    else:
                        res[f, k_x, k_y] *= 0.8
                else:
                    res[f, k_x, k_y] /= 2
    return res


def _mark_one_harm(data: np.ndarray, df: int, turn_period: float, k_min: float, k_max: float, speed: float,
                   angle: float, inverse=False, flag_speed=1):
    """
    marking main harmonics \omega = \sqrt{gk} + kV
    @param data: input data after fourier transform with background and signal
    @param df: window of frequency for cutting
    @param turn_period: period of one radar's turn
    @param k_min: minimum of wave number
    @param k_max: maximum of wave number
    @param speed: speed of vessel in meters/sec
    @param angle: angle between speed and wave vector
    @param inverse: True if needed to mirror input array
    @param flag_speed: forward/backward flag
    return input array with zeroes on elements in dispersion relation
    """
    res = np.copy(data)
    for k_x in range(data.shape[1]):
        for k_y in range(data.shape[2]):
            alpha = np.arctan2(k_y, k_x) + np.pi * angle / 180  # + np.pi / 4
            # convert from wave vector in pix to wave number in rad/meters
            k_abs = np.sqrt(k_x ** 2 + k_y ** 2) / data.shape[1] * k_max + k_min
            # calculating a frequency by dispersion relation
            freq = np.sqrt(9.81 * k_abs) - flag_speed * k_abs * speed * np.cos(alpha)

            freq /= (2 * np.pi)
            # convert in index
            freq *= (turn_period * 256)
            freq = int(freq)

            if inverse:
                freq -= df

            left = min(max(freq - df, 0), 255)
            right = min(freq + df + 1, 255)

            res[left, k_x, k_y] = 0
            res[right, k_x, k_y] = 0

    return res



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

    popt, pcov = curve_fit(dispersion_func, np.linspace(0, k_max, arr_2d.shape[1])[mask],
                           max_freq[mask] / arr_2d.shape[0] / turn_period,
                           sigma=1 / np.max(arr_2d, axis=0)[mask], absolute_sigma=True)

    axs.scatter(np.linspace(0, k_max, arr_2d.shape[1]), max_freq / arr_2d.shape[0] / turn_period, s=2)
    axs.scatter(np.linspace(0, k_max, arr_2d.shape[1])[mask], max_freq[mask] / arr_2d.shape[0] / turn_period, s=2)
    axs.plot(np.linspace(0, k_max, arr_2d.shape[1]), dispersion_func(np.linspace(0, k_max, arr_2d.shape[1]), 0),
            ls='--')
    axs.plot(np.linspace(0, k_max, arr_2d.shape[1]), dispersion_func(np.linspace(0, k_max, arr_2d.shape[1]), *popt),
            ls='--')

    noise = np.copy(arr_2d)
    l = []
    r = []

    for k in range(0, arr_2d.shape[1]):

        if k > 0:
            arr_2d[:, k] *= (k ** (-1.15))
            noise[:, k] *= (k ** (-1.15))

        freq = dispersion_func(k / arr_2d.shape[1] * k_max, *popt) * 256 * turn_period
        # df_new = df * np.log(np.max(arr_2d[:, k])) / np.log(np.max(arr_2d))
        if (freq > arr_2d.shape[0]) or (k > arr_2d.shape[1] - 10):
            df_new = 1
        else:
            df_new = df * np.log(arr_2d[int(freq), k]) / np.log(np.max(arr_2d))
        #
        if (freq > 2 * df) and (freq < arr_2d.shape[0] - 2 * df):
            df_new = df * (np.log(arr_2d[round(freq), k]) - np.log(
                np.min(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, k]))) / (
                             np.log(np.max(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, :])) - np.log(
                                np.min(arr_2d[round(freq) - 2 * df: round(freq) + 2 * df, k])))

        left = max(round(freq - df_new), 0)
        l.append(left)
        right = min(round(freq + df_new + 1), 256)
        r.append(right)

        if right > left:
            noise[left: right, k] = np.zeros(right - left)

        # for f in range(left, right):
        #   dist_f = np.abs(round(f - freq - df_new // 2))
        #   # special coefficients for smooth cutting
        #   if dist_f <= df_new / 3:
        #       noise[f, k] = 0
        #   elif dist_f <= 7 * df_new / 9:
        #       noise[f, k] *= 0.3
        #   else:
        #       noise[f, k] *= 0.8

    axs.plot(np.linspace(0, k_max, 32), np.array(l) / 256 / turn_period, color='white', linewidth=1)
    axs.plot(np.linspace(0, k_max, 32), np.array(r) / 256 / turn_period, color='white', linewidth=1)
    plt.savefig(PATH + "pics/" + name[-7:-3] + ".png", bbox_inches='tight', dpi=700)
    plt.show()

    int_ind_k = 0
    int_ind_f = 0

    signal = arr_2d - noise

    ss = trapezoid(signal[int_ind_k:])
    nn = trapezoid(noise[int_ind_k:])

    freq_specter = ss / nn
    m0 = trapezoid(ss[int_ind_f:]) / trapezoid(nn[int_ind_f:])
    m1 = trapezoid(ss[int_ind_f:] * np.linspace(0, 1 / turn_period, ss[int_ind_f:].shape[0])) / \
         trapezoid(nn[int_ind_f:] * np.linspace(0, 1 / turn_period, ss[int_ind_f:].shape[0]))

    return m0, m1, freq_specter, np.arccos(max(min(popt[0] / speed, 1), -1)) / np.pi * 180


"""
def calc_forward_toward(array: np.ndarray, window: int):

    determine direction of wave in current packet
    :param array: input 2d array of data after Radon transform cut in current direction
    :param window: if AN is found peak, then in [AN - window, AN + window] new peaks won't find

    toward = np.zeros(array.shape[0], dtype=int)
    copy = np.copy(array[0])
    SIZE = 5  # number of peaks
    c_win = 2  # coefficient in window for zeroing
    argmax_old = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        argmax_old[i] = np.argmax(copy)
        _zero_array(copy, argmax_old[i], c_win * window)

    for t in range(1, array.shape[0]):

        copy = np.copy(array[t])
        argmax_new = np.zeros(SIZE, dtype=int)
        for i in range(SIZE):
            argmax_new[i] = np.argmax(
                copy[max(0, argmax_old[i] - window): min(argmax_old[i] + window, len(copy))]) - window

            if argmax_new[i] >= 0:
                toward[t] += 1
            else:
                toward[t] -= 1

        for i in range(SIZE):
            argmax_old[i] = np.argmax(copy)
            _zero_array(copy, argmax_old[i], c_win * window)

    print(toward)
    print(np.abs(np.mean(toward)) / SIZE)

    if np.abs(np.mean(toward)) / SIZE <= 0.1:
        return False, False

    return True, np.sum(toward) >= 0
"""
def process_fourier(name, array: np.ndarray, speed: float, angle: float, df: int, turn_period: float, k_min: float,
                    k_max: float):
    """
    calculate specter from 3D cartesian zone
    @param array: input 3D data
    @param turn_period: period of one radar's turn
    @param df: window of frequency for cutting
    @param k_min: minimum of wave number
    @param k_max: maximum of wave number
    @param speed: speed of vessel in meters/sec
    @param angle: angle between speed and wave vector
    """
    trap = np.trapz(array[:, 10, :])

    # plt.plot(trap)
    # plt.show()

    if np.sum(trap[array.shape[0] // 5:array.shape[0] // 2]) > np.sum(trap[array.shape[0] // 2:- array.shape[0] // 5]):

        data_bgn1 = cut_one_harm_3d(array, df, turn_period, k_min, k_max, speed, angle, False, 1)
        data_bgn2 = cut_one_harm_3d(array, df, turn_period, k_min, k_max, speed, angle, False, -1)

        mark_disp1 = _mark_one_harm(array, df, turn_period, k_min, k_max, speed, angle, False, 1)
        mark_disp2 = _mark_one_harm(array, df, turn_period, k_min, k_max, speed, angle, False, -1)

        if np.sum(data_bgn1) < np.sum(data_bgn2):
            data_bgn = data_bgn1
            mark_disp = mark_disp1
        else:
            data_bgn = data_bgn2
            mark_disp = mark_disp2
        data_disp = array - data_bgn

    else:
        data_bgn1 = cut_one_harm_3d(np.flip(array, 0), df, turn_period, k_min, k_max, speed, angle, True, 1)
        data_bgn2 = cut_one_harm_3d(np.flip(array, 0), df, turn_period, k_min, k_max, speed, angle, True, -1)

        mark_disp1 = _mark_one_harm(array, df, turn_period, k_min, k_max, speed, angle, True, 1)
        mark_disp2 = _mark_one_harm(array, df, turn_period, k_min, k_max, speed, angle, True, -1)

        if np.sum(data_bgn1) > np.sum(data_bgn2):
            data_bgn = data_bgn1
            mark_disp = mark_disp1
        else:
            data_bgn = data_bgn2
            mark_disp = mark_disp2

        data_disp = np.flip(array, 0) - data_bgn

    make_anim_four(mark_disp, "mark_disp_" + str(name[-7:-3]))
    # make_anim_four(array, "or_2_" + str(name[-17:-3]))
    # make_anim_four(data_bgn, "bgn_2_" + str(name[-17:-3]))
    # make_anim_four(data_disp, "disp_2_" + str(name[-17:-3]))

    for x in range(1, data_disp.shape[0]):
        for y in range(1, data_disp.shape[1]):
            data_disp[x, y] *= (x ** 2 + y ** 2) ** (-0.55)

    int_ind_f = 2
    int_ind_k = 2

    freq = trapezoid(trapezoid(data_disp[int_ind_k:])[int_ind_k:]) / trapezoid(
        trapezoid(data_bgn[int_ind_k:])[int_ind_k:])

    m0 = trapezoid(trapezoid(trapezoid(data_disp[int_ind_k:])[int_ind_k:])[int_ind_f:]) / trapezoid(
        trapezoid(trapezoid(data_bgn[int_ind_k:])[int_ind_k:])[int_ind_f:])

    m1 = 0

    return m0, m1, freq