import numpy as np
from numba import njit


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


def find_main_directions(array, window: int, num_peaks: int):
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
            copy[angles[i] - window: -1] = np.zeros_like(copy[angles[i] - window: -1])
            copy[0: angles[i]] = np.zeros_like(copy[0: angles[i]])
        else:
            # print(angles[i] - window, angles[i])
            copy[angles[i] - window: angles[i]] = np.zeros_like(copy[angles[i] - window: angles[i]])
        if angles[i] + window > copy.shape[0]:
            copy[angles[i]: -1] = np.zeros_like(copy[angles[i]: -1])
            copy[0: angles[i] + window - copy.shape[0]] = np.zeros_like(copy[0: angles[i] + window - copy.shape[0]])
        else:
            copy[angles[i]: angles[i] + window] = np.zeros_like(copy[angles[i]: angles[i] + window])

        if i != 0:
            if np.abs(angles[i] - angles[i - 1]) % 360 < 2 * window:
                print("pop")
                angles.pop()

    return angles


def calc_forward_toward(array, window):
    toward = np.zeros(array.shape[0], dtype=int)
    copy = np.copy(array[0])
    SIZE = 4
    argmax_old = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        argmax_old[i] = np.argmax(copy)
        copy[argmax_old[i] - 5 * window: argmax_old[i] + 5 * window] = np.zeros(10 * window)

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
            copy[argmax_old[i] - 5 * window: argmax_old[i] + 5 * window] = np.zeros(10 * window)

    print(toward)

    return np.sum(toward) >= 0
