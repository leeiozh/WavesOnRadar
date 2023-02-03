import time
import glob

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import scipy.signal as ss
import scipy.fft as sf
from scipy import ndimage, misc
import pandas as pd
from skimage.transform import radon

from drawers import *
from structures import *
from calculators import *

ask = input("manually enter stations or just enter for auto process all stations >> ")

if len(ask) < 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))
elif len(ask) == 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))[:int(ask)]
else:
    stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*' + str(f) + '*.nc'))[0] for f in ask.split()]

stations.sort()

cut_ind = 32
time_radon = 64
time_fourier = 256
PERIOD_RADAR = 2.5
resolution = 4096
TRASHHOLD = 0.5
SIZE_SQUARE_PIX = 384
k_min = 2 * np.pi / 720
k_max = 2 * np.pi / 720 * cut_ind

an_res = []

mask_circle = make_radon_circle(SIZE_SQUARE_PIX)

for name in stations:
    data_nc = nc.Dataset(name)  # имя файла с данными
    log_file = pd.read_csv("sheets/stations_data10.csv", delimiter=",")

    print("station " + name[-7:-3] + " started proccess...")

    max_time = data_nc.variables["bsktr_radar"].shape[0]

    if max_time < time_fourier:
        print("!!!!!!!!!!!!!!!!!")
        print(name, "is short, max_time", max_time)
        print("!!!!!!!!!!!!!!!!!")

    if max_time < time_fourier // 2:
        break

    back_cartesian_3d_rad = np.ndarray(shape=(time_fourier, SIZE_SQUARE_PIX, SIZE_SQUARE_PIX), dtype=float)
    # back_cartesian_3d_four_one = np.ndarray(shape=(SIZE_SQUARE_PIX, SIZE_SQUARE_PIX), dtype=float)
    radon_array = np.ndarray(shape=(time_fourier, SIZE_SQUARE_PIX, 180), dtype=float)
    radon_max = np.ndarray(shape=(time_fourier,), dtype=float)

    time0 = time.time()

    for t in range(time_radon):

        if t % 8 == 0 and t != 0:
            print("radon done " + str(round(t / time_radon * 100, 1)) + "%, estimated time " + str(
                round((time_radon / t - 1) * (time.time() - time0), 1)))

        area = Area(720, 720, 1360, 270, -270)  # int(data_nc.variables["giro_radar"][t])

        area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
                                                               data_nc.variables["rad_radar"].shape[0],
                                                               data_nc.variables["theta_radar"].shape[0], resolution)

        if t < max_time:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][t])
        else:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t + 2 * max_time - 1])

        back_cartesian_3d_rad[t] = speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod)
        back_cartesian_3d_rad[t][back_cartesian_3d_rad[t] < 2 * TRASHHOLD * np.mean(back_cartesian_3d_rad[t])] = 0.
        back_cartesian_3d_rad[t][back_cartesian_3d_rad[t] != 0.] = 1.

        radon_array[t] = radon(mask_circle * back_cartesian_3d_rad[t])
        # radon_array[t] = np.roll(radon_array[t], 270, axis=-1)  # int(data_nc.variables["giro_radar"][t])

        # tmp_1 = ndimage.sobel(back_cartesian_3d_four_one, axis=1, mode='nearest')
        # tmp_0 = ndimage.sobel(back_cartesian_3d_four_one, axis=0, mode='nearest')
        # back_cartesian_3d_four_one = np.hypot(tmp_0, tmp_1)

    # freq, res_s = ss.welch(back_cartesian_3d_four, detrend='linear', axis=0, return_onesided=False)

    # make_anim_back(back_cartesian_3d_four, full_time, str("back1" + name[-7:-3]))
    # make_anim_radon(radon_array, full_time, str("radon1" + name[-7:-3]))

    angles = find_main_directions(radon_array, 1, 15, 5)

    print("radon done, obtained directions >> ", angles)

    for an in angles:

        back_cartesian_3d_four = np.ndarray(shape=(time_fourier, cut_ind, cut_ind), dtype=complex)

        for t in range(time_fourier):
            if t % 16 == 0 and t != 0:
                print("fourier done " + str(round(t / time_fourier * 100, 1)) + "%, estimated time " + str(
                    round((time_fourier / t - 1) * (time.time() - time0), 1)))

            area = Area(720, 720, 1360, 270, -270 - an)  # int(data_nc.variables["giro_radar"][t])

            area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
                                                                   data_nc.variables["rad_radar"].shape[0],
                                                                   data_nc.variables["theta_radar"].shape[0],
                                                                   resolution)

            if t < max_time:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][t])
            else:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t + 2 * max_time - 1])

            back_cartesian_3d_four[t] = sf.fft2(speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod))[
                                        :cut_ind, :cut_ind]

        f, res_s = ss.welch(back_cartesian_3d_four, detrend='linear', axis=0, return_onesided=False)
        # make_anim_back(res_s, str("back" + name[-7:-3]))
        # trap = np.trapz(res_s[:, 10, :])

        m0, m1, freq = process_fourier(res_s, np.median(data_nc.variables["sog_radar"]),
                                       an + np.median(data_nc.variables["giro_radar"]), 10, PERIOD_RADAR, k_min, k_max)

        plt.plot(freq)
        plt.savefig("pics/freq_" + str(name[-7:-3]) + ".png")
        plt.close()

    log_file.loc[log_file["name"] == int(name[-7:-3]), ["an"]] = angles[0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["m0"]] = m0

    if len(angles) > 1:
        log_file.loc[log_file["name"] == int(name[-7:-3]), ["an2"]] = angles[1]

    log_file.to_csv("sheets/stations_data10.csv", index=False)

    print("station " + name[-7:-3] + " processed and saved")
    data_nc.close()
