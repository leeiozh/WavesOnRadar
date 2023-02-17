import statsmodels.api as sm
from skimage.transform import radon
from scipy.integrate import trapezoid
import scipy.fft as sf
from drawers import *
from structures import *


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
def make_radon_circle(size: int):
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


def find_main_directions(radon_array: np.ndarray, num_peaks: int, window: int, one: bool):
    """
    search num_peaks (or less) main wave direction
    :param radon_array: input 3d array result of Radon transform
    :param num_peaks: maximal number of directions in loop
    :param window: if AN is found direction, then in [AN - window, AN + window] new direction won't find
    :param one: flag for maximum two starts this function
    """

    copy = np.mean(np.std(radon_array, axis=1), axis=0)

    grad = np.gradient(copy)
    angles = []
    for i in range(num_peaks):
        if i == 0:
            angles.append(np.argmax(copy))
        else:
            an = np.argmax(copy)
            tmp = np.diff(np.sign(grad[min(an, angles[-1]): max(an, angles[-1])]))
            if np.sum(np.abs(tmp[tmp != 0.])) != 0:
                angles.append(an)
            else:
                break

        _zero_array(copy, angles[i], window)

        if i != 0:
            if np.abs(angles[i] - angles[i - 1]) % 360 < 2 * window:
                angles.pop()

    for i in range(len(angles)):

        flag, direct = calc_forward_toward(radon_array[:, :, angles[i]], 10)

        # direct = calc_autocorr(radon_array[:, :, angles[i]])
        # if direct != -1:

        if flag:
            if direct:
                angles[i] *= -1
                angles[i] += 360
            if angles[i] > 360:
                angles[i] -= 360
            if angles[i] < 0:
                angles[i] += 360

        elif one:  # if in first time we can't find a direction
            return 666

    return angles


def calc_autocorr(array: np.ndarray):
    # эта штука еще не допилена, из-за нее могут получаться плохие результаты
    ans = np.zeros(array.shape[0] - 1)

    for t in range(ans.shape[0]):
        if np.correlate(array[t], np.roll(array[t + 1], 15))[0] < np.correlate(array[t], np.roll(array[t + 1], -15))[0]:
            ans[t] = 1

    if np.abs(np.sum(ans) / ans.shape[0] - 0.5) < 0.05:
        return -1

    return np.sum(ans) / ans.shape[0] >= 0.5


def calc_forward_toward(array: np.ndarray, window: int):
    """
    determine direction of wave in current packet
    :param array: input 2d array of data after Radon transform cut in current direction
    :param window: if AN is found peak, then in [AN - window, AN + window] new peaks won't find
    """
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


def _zero_array(array, ind, window):
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


def diff_one_harm_3d(data, df, turn_period, k_min, k_max, speed, angle, inverse=False, flag_speed=1):
    res = np.copy(data)
    for k_x in range(data.shape[1]):
        for k_y in range(data.shape[2]):
            alpha = np.arctan2(k_y, k_x) + np.pi * angle / 180  # + np.pi / 4
            # перевод волнового вектора из индексной размерности в модуль рад/м
            k_abs = np.sqrt(k_x ** 2 + k_y ** 2) / data.shape[1] * k_max + k_min
            # вычисление соответствующей частоты из дисперсионного соотношения
            freq = np.sqrt(9.81 * k_abs) + flag_speed * k_abs * speed * np.cos(alpha)

            freq /= (2 * np.pi)
            # перевод в индекс
            freq *= (turn_period * 256)
            freq = int(freq)

            if inverse:
                freq -= df

            left = max(freq - df, 0)
            right = min(freq + df + 1, 256)

            for f in range(left, right):
                if f > 127 + 0.25 * df or f < 127 - 0.25 * df:
                    dist_f = np.abs(f - freq - df // 2)
                    if dist_f <= df / 3:
                        res[f, k_x, k_y] = 0
                    elif dist_f <= 7 * df / 9:
                        res[f, k_x, k_y] *= 0.3
                    else:
                        res[f, k_x, k_y] *= 0.8
                else:
                    res[f, k_x, k_y] /= 2
    return res


def thresh_radon(array, size_array, threshold, mask_circle):
    radon_array = np.ndarray(shape=(array.shape[0], size_array, 180), dtype=float)
    for t in range(array.shape[0]):
        array[t][array[t] < 2 * threshold * np.mean(array[t])] = 0
        array[t][array[t] != 0] = 1
        radon_array[t] = radon(mask_circle * array[t])
    return radon_array


def calc_std(data_nc, resolution, start_ind, version):
    # area = Area(3450, 3450, 0, 0, 0)
    # if version:
    #    area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
    #                                                           data_nc.variables["rad_radar"].shape[0],
    #                                                           data_nc.variables["theta_radar"].shape[0], resolution)
    # else:
    #    area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["radar_rad"][-1],
    #                                                           data_nc.variables["radar_rad"].shape[0],
    #                                                           data_nc.variables["radar_theta"].shape[0], resolution)
    back_center = np.ndarray(shape=(4, 4096, 1024), dtype=float)
    zero_rho_back = 512
    for t in range(back_center.shape[0]):
        if version:
            # back_center[t] = speed_bilinear_interpol(np.transpose(data_nc.variables["bsktr_radar"][t + start_ind]),
            #                                          area_mask_div, area_mask_mod)
            back_center[t] = data_nc.variables["bsktr_radar"][t + start_ind][:,
                             zero_rho_back:zero_rho_back + back_center.shape[2]]
        else:
            # back_center[t] = speed_bilinear_interpol(np.transpose(data_nc.variables["bsktr"][t + start_ind]),
            #                                          area_mask_div, area_mask_mod)
            back_center[t] = data_nc.variables["bsktr"][t + start_ind][:,
                             zero_rho_back:zero_rho_back + back_center.shape[2]]

    std_array = np.std(back_center, axis=0)
    # plt.imshow(std_array, cmap='Greys', origin='lower')
    # plt.show()

    argmax = np.argmax(std_array)

    theta = (argmax // std_array.shape[1]) / std_array.shape[0] * 360
    rho = (argmax % std_array.shape[1] + zero_rho_back) * 1.875

    return theta, rho


def calc_back(data_nc, calc_time, start_ind, max_time, resolution, area_size, size_array, flag_area, an, version):
    if flag_area != 2:
        back = np.ndarray(shape=(calc_time, size_array, size_array), dtype=float)
    else:
        back = np.ndarray(shape=(calc_time, size_array, size_array), dtype=complex)

    for t in range(calc_time):

        if t % 120 == 0 and flag_area != 1:
            angle_std, rad_std = calc_std(data_nc, resolution, start_ind + t, version)

            if np.abs(angle_std - 6) < 6:
                angle_std += np.sign(angle_std - 6) * 2 * np.abs(angle_std - 6)

        # if t % 32 == 0 and t != 0:
        #     print("back done " + str(round(t / calc_time * 100, 1)) + "%, estimated time " + str(
        #         round((calc_time / t - 1) * (time.time() - time0), 1)))

        if flag_area == 0:
            zone = Area(area_size, area_size, rad_std, angle_std, 0)
        if flag_area == 1:
            zone = Area(area_size, area_size, area_size * 2, 270, -270)
        if flag_area == 2:
            zone = Area(area_size, area_size, rad_std, angle_std, (-angle_std - an) % 360)

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

    return back


def process_fourier(array, speed, angle, WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max):
    trap = np.trapz(array[:, 10, :])

    if np.sum(trap[array.shape[0] // 5:array.shape[0] // 2]) > np.sum(trap[array.shape[0] // 2:- array.shape[0] // 5]):

        data_bgn1 = diff_one_harm_3d(array, WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max, speed, angle, False, 1)
        data_bgn2 = diff_one_harm_3d(array, WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max, speed, angle, False, -1)

        if np.sum(data_bgn1) < np.sum(data_bgn2):
            data_bgn = data_bgn1
        else:
            data_bgn = data_bgn2
        data_disp = array - data_bgn

    else:
        data_bgn1 = diff_one_harm_3d(np.flip(array, 0), WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max, speed, angle,
                                     True, 1)
        data_bgn2 = diff_one_harm_3d(np.flip(array, 0), WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max, speed, angle,
                                     True, -1)
        if np.sum(data_bgn1) > np.sum(data_bgn2):
            data_bgn = data_bgn1
        else:
            data_bgn = data_bgn2
        data_disp = np.flip(array, 0) - data_bgn

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
