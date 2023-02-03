import numpy as np
from numba import njit
from scipy.integrate import trapezoid


@njit
def speed_bilinear_interpol(back, mask_div, mask_mod):
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
def speed_no_interpol(back, mask_div):
    res = np.zeros((mask_div.shape[1], mask_div.shape[2]))
    for i in range(mask_div.shape[1]):
        for j in range(mask_div.shape[2]):
            res[i, j] = back[mask_div[0, i, j], mask_div[1, i, j]]
    return res


def make_radon_circle(size):
    res = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if (x - size // 2) * (x - size // 2) + (y - size // 2) * (y - size // 2) < (size // 2) * (size // 2):
                res[x, y] = 1.
    return res


def find_main_directions(radon_array, num_peaks: int, window: int, window_for: int):
    array = np.mean(np.std(radon_array, axis=1), axis=0)

    copy = np.copy(array)
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
        if angles[i] - window < 0:
            copy[angles[i] - window: -1] = np.zeros(copy.shape[0] - angles[i] - window)
            copy[0: angles[i]] = np.zeros(angles[i])
        else:
            # print(angles[i] - window, angles[i])
            copy[angles[i] - window: angles[i]] = np.zeros(window)
        if angles[i] + window > copy.shape[0]:
            copy[angles[i]: -1] = np.zeros(copy.shape[0] - angles[i])
            copy[0: angles[i] + window - copy.shape[0]] = np.zeros(angles[i] + window - copy.shape[0])
        else:
            copy[angles[i]: angles[i] + window] = np.zeros(window)

        if i != 0:
            if np.abs(angles[i] - angles[i - 1]) % 360 < 2 * window:
                angles.pop()

    for i in range(len(angles)):
        flag = calc_forward_toward(radon_array[:, :, angles[i]], window_for)
        if flag:
            angles[i] += 180
        if angles[i] > 360:
            angles[i] -= 360
        if angles[i] < 0:
            angles[i] += 360

    return angles


def calc_forward_toward(array, window):
    toward = np.zeros(array.shape[0], dtype=int)
    copy = np.copy(array[0])
    SIZE = 5
    coeff_win = 2
    argmax_old = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        argmax_old[i] = np.argmax(copy)
        # copy[argmax_old[i] - 5 * window: argmax_old[i] + 5 * window] = np.zeros(10 * window)

        if argmax_old[i] - coeff_win * window < 0:
            copy[argmax_old[i] - coeff_win * window: -1] = np.zeros(-argmax_old[i] + coeff_win * window - 1)
            copy[0:argmax_old[i]] = np.zeros(argmax_old[i])
        else:
            # print(angles[i] - window, angles[i])
            copy[argmax_old[i] - coeff_win * window: argmax_old[i]] = np.zeros(coeff_win * window)
        if argmax_old[i] + coeff_win * window > copy.shape[0]:
            copy[argmax_old[i]: -1] = np.zeros(copy.shape[0] - argmax_old[i])
            copy[0: argmax_old[i] + coeff_win * window - copy.shape[0]] = np.zeros(
                argmax_old[i] + coeff_win * window - copy.shape[0])
        else:
            copy[argmax_old[i]: argmax_old[i] + coeff_win * window] = np.zeros(coeff_win * window)

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
            # copy[argmax_old[i] - 5 * window: argmax_old[i] + 5 * window] = np.zeros(10 * window)
            if argmax_old[i] - coeff_win * window < 0:
                copy[argmax_old[i] - coeff_win * window: -1] = np.zeros(- argmax_old[i] + coeff_win * window - 1)
                copy[0:argmax_old[i]] = np.zeros(argmax_old[i])
            else:
                # print(angles[i] - window, angles[i])
                copy[argmax_old[i] - coeff_win * window: argmax_old[i]] = np.zeros(coeff_win * window)
            if argmax_old[i] + coeff_win * window > copy.shape[0]:
                copy[argmax_old[i]: -1] = np.zeros(copy.shape[0] - argmax_old[i])
                copy[0: argmax_old[i] + coeff_win * window - copy.shape[0]] = np.zeros(
                    argmax_old[i] + coeff_win * window - copy.shape[0])
            else:
                copy[argmax_old[i]: argmax_old[i] + coeff_win * window] = np.zeros(coeff_win * window)

    return np.sum(toward) >= 0


def diff_one_harm_3d(data, df, turn_period, k_min, k_max, speed, angle, inverse=False, flag_speed=1):
    res = np.copy(data)
    for k_x in range(data.shape[1]):
        for k_y in range(data.shape[2]):
            alpha = np.arctan2(k_y, k_x) + np.pi * angle / 180  # + np.pi / 4
            # перевод волнового вектора из индексной размерности в модуль рад/м
            k_abs = np.sqrt(k_x ** 2 + k_y ** 2) / data.shape[1] * k_max + k_min
            # вычисление соответствующей частоты из дисперсионного соотношения
            freq = np.sqrt(9.81 * k_abs) + flag_speed * k_abs * speed * np.cos(alpha)

            # freq = np.sqrt(9.81 * k_abs) + k_abs * speed * (np.sin(alpha) - 1.1)

            # if k_abs < 0.05 * k_max:
            #     print(round(np.sqrt(9.81 * k_abs), 2), round(k_abs * speed * np.cos(alpha), 2))

            freq /= (2 * np.pi)
            # перевод в индекс
            freq *= (turn_period * 256)
            freq = int(freq)
            if not inverse:
                for f in range(max(freq - df, 0), min(freq + df + 1, 256)):
                    if f > 127 + 0.75 * df or f < 127 - 0.75 * df:
                        res[f, k_x, k_y] = 0
                    else:
                        res[f, k_x, k_y] /= 10
            if inverse:
                freq -= df
                # freq = 256 - freq
                for f in range(max(freq - df, 0), min(freq + df + 1, 256)):
                    if f > 127 + 0.75 * df or f < 127 - 0.75 * df:
                        res[f, k_x, k_y] = 0
                    else:
                        res[f, k_x, k_y] /= 10
                        # if inverse and (freq > 127 + df or freq < 127 - df):
            #     freq = 256 - freq
            #     for f in range(max(freq - df, 0), min(freq + df + 1, 256)):
            #         res[f, k_x, k_y] = 0
            # res[:, k_x, k_y] *= (k_abs / k_max) ** (-1.2)
    return res


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
        data_bgn1 = diff_one_harm_3d(np.flip(array, 0), WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max, speed, True, 1)
        data_bgn2 = diff_one_harm_3d(np.flip(array, 0), WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max, speed, True, -1)
        if np.sum(data_bgn1) > np.sum(data_bgn2):
            data_bgn = data_bgn1
        else:
            data_bgn = data_bgn2
        data_disp = np.flip(array, 0) - data_bgn

    for x in range(1, data_disp.shape[0]):
        for y in range(1, data_disp.shape[1]):
            data_disp[x, y] *= (x ** 2 + y ** 2) ** (-0.55)

    int_ind_f = 0
    int_ind_k = 0

    freq = trapezoid(trapezoid(data_disp[int_ind_k:])[int_ind_k:]) / trapezoid(
        trapezoid(data_bgn[int_ind_k:])[int_ind_k:])

    m0 = trapezoid(trapezoid(trapezoid(data_disp[int_ind_k:])[int_ind_k:])[int_ind_f:]) / trapezoid(
        trapezoid(trapezoid(data_bgn[int_ind_k:])[int_ind_k:])[int_ind_f:])

    m1 = trapezoid(
        trapezoid(trapezoid(data_disp[int_ind_k:])[int_ind_k:])[int_ind_f:] * np.linspace(0, 1 / PERIOD_RADAR,
                                                                                          data_disp.shape[
                                                                                              0] - int_ind_f)) / m0

    return m0, m1, freq
