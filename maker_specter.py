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
full_time = 64
PERIOD_RADAR = 2.5
resolution = 4096
TRASHHOLD = 0.5

an_res = []

mask_circle = make_radon_circle(384)

for name in stations:
    data_nc = nc.Dataset(name)  # имя файла с данными
    log_file = pd.read_csv("sheets/stations_data10.csv", delimiter=",")

    print("station " + name[-7:-3] + " started proccess...")

    max_time = data_nc.variables["bsktr_radar"].shape[0]

    if max_time < full_time:
        print("!!!!!!!!!!!!!!!!!")
        print(name, "is short, max_time", max_time)
        print("!!!!!!!!!!!!!!!!!")

    if max_time < full_time // 2:
        break

    back_cartesian_3d_four = np.ndarray(shape=(full_time, 384, 384), dtype=float)
    # back_cartesian_3d_four_one = np.ndarray(shape=(384, 384), dtype=float)
    radon_array = np.ndarray(shape=(full_time, 384, 180), dtype=float)
    radon_max = np.ndarray(shape=(full_time,), dtype=float)

    time0 = time.time()

    for t in range(full_time):

        if t % 13 == 0 and t != 0:
            print("done " + str(round(t / full_time * 100, 1)) + "%, estimated time " + str(
                round((full_time / t - 1) * (time.time() - time0), 1)))

        area = Area(720, 720, 1360, 270, 0)

        area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
                                                               data_nc.variables["rad_radar"].shape[0],
                                                               data_nc.variables["theta_radar"].shape[0], resolution)

        if t < max_time:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][t])
        else:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t + 2 * max_time - 1])

        back_cartesian_3d_four[t] = speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod)
        back_cartesian_3d_four[t][back_cartesian_3d_four[t] < 2 * TRASHHOLD * np.mean(back_cartesian_3d_four[t])] = 0.
        back_cartesian_3d_four[t][back_cartesian_3d_four[t] != 0.] = 1.

        radon_array[t] = radon(mask_circle * back_cartesian_3d_four[t])

        plt.plot(np.std(radon_array[t], axis=0), color='blue')

        # tmp_1 = ndimage.sobel(back_cartesian_3d_four_one, axis=1, mode='nearest')
        # tmp_0 = ndimage.sobel(back_cartesian_3d_four_one, axis=0, mode='nearest')
        # back_cartesian_3d_four_one = np.hypot(tmp_0, tmp_1)

    # freq, res_s = ss.welch(back_cartesian_3d_four, detrend='linear', axis=0, return_onesided=False)

    # make_anim_back(back_cartesian_3d_four, full_time, str("nosobel_trash05" + name[-7:-3]))
    # make_anim_radon(radon_array, full_time, str("radon_trash05" + name[-7:-3]))

    men = np.mean(np.std(radon_array, axis=1), axis=0)

    angles = find_main_directions(men, 20, 2)

    plt.plot(men, color='red')

    for an in angles:
        plt.plot([an, an], [np.min(men), np.max(men)], color='green')
        flag = calc_forward_toward(radon_array[:, :, an], 3)

        if not flag:
            print("toward")
            an -= 90
        else:
            print("forward")
            an += 90

        if an > 360:
            an -= 360
        if an < 0:
            an += 360

        print(an)
        an_res.append(an)

    log_file.loc[log_file["name"] == int(name[-7:-3]), ["sog"]] = data_nc.variables["sog_radar"][0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["cog"]] = data_nc.variables["cog_radar"][0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["giro"]] = data_nc.variables["giro_radar"][0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["an"]] = an_res[-2]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["an2"]] = an_res[-1]
    # log_file.loc[log_file["name"] == int(name[-7:-3]), ["angle_radon_nosob2"]] = np.median(radon_max)
    ## # #
    log_file.to_csv("sheets/stations_data10.csv", index=False)

    print("station " + name[-7:-3] + " processed and saved")
    # print(np.median(np.argmax(std, axis=1)))

    data_nc.close()
