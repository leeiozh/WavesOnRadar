import time
import glob

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import scipy.signal as ss
import scipy.fft as sf
import pandas as pd
from skimage.transform import radon

from drawers import *
from structures import *
from calculators import *

ask = int(input("press 0 for manual input stations or 1 for auto process all stations >> "))

if ask != 0:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))
else:
    stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*' + str(f) + '*.nc'))[0] for f in
                input("enter stations >> ").split()]

stations.sort()

cut_ind = 64
full_time = 256
PERIOD_RADAR = 2.5
resolution = 4096

mask_circle = make_radon_circle(384)

for name in stations:
    data_nc = nc.Dataset(name)  # имя файла с данными
    # log_file = pd.read_csv("stations_data1.csv", delimiter=",")

    print("station " + name[-7:-3] + " started proccess...")

    max_time = data_nc.variables["bsktr_radar"].shape[0]

    if max_time < full_time:
        print("!!!!!!!!!!!!!!!!!")
        print(name, "is short, max_time", max_time)
        print("!!!!!!!!!!!!!!!!!")

    if max_time < full_time // 2:
        break

    area = Area(720, 720, 1360, 270, 0)

    area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
                                                           data_nc.variables["rad_radar"].shape[0],
                                                           data_nc.variables["theta_radar"].shape[0], resolution)

    back_cartesian_3d_four = np.ndarray(shape=(full_time, 384, 384), dtype=float)
    radon_array = np.ndarray(shape=(full_time, 384, 180), dtype=float)
    radon_argmax = np.ndarray(shape=(full_time,), dtype=float)
    radon_std = np.ndarray(shape=(full_time,), dtype=float)

    time0 = time.time()

    for t in range(full_time):

        if t % 13 == 0 and t != 0:
            print("done " + str(round(t / full_time * 100, 1)) + "%, estimated time " + str(
                round((full_time / t - 1) * (time.time() - time0), 1)))

        if t < max_time:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][t])
        else:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t + 2 * max_time - 1])

        back_cartesian_3d_four[t] = speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod)

        radon_array[t] = radon(mask_circle * back_cartesian_3d_four[t])
#
        #radon_argmax[t] = np.argmax(np.max(radon_array[t], axis=0))
        #radon_std[t] = np.argmax(np.std(radon_array[t], axis=0))

        # plt.imshow(radon(mask_circle * back_cartesian_3d_0), origin='lower', cmap='gnuplot', aspect=0.5)
        # plt.savefig("radon" + str(t) + ".png")

        # print(radon_argmax[t], end=' ')
        # back_cartesian_3d_four[t] = sf.fft2(back_cartesian_3d_0)[:cut_ind, :cut_ind]

    # freq, res_s = ss.welch(radon_array, detrend='linear', axis=0, return_onesided=False)
    # make_shot(res_s[:, :, int(np.mean(radon_argmax))], str("radon_max" + name[-7:-3]))
    # make_shot(res_s[:, :, int(np.mean(radon_std))], str("radon_std" + name[-7:-3]))

    # plt.plot(radon_std)
    # plt.plot(radon_argmax)
    # plt.savefig("radon_tmp" + str(name[-7:-3]) + ".png")
    #
    # print("fourier done, start saving ...")

    # res_resh = res_s.reshape(res_s.shape[0], -1)
    # radon_array_resh = radon_array.reshape(radon_array.shape[0], -1)
    # np.savetxt('results/radon_256_' + name[-7:-3] + '.csv', radon_array_resh, delimiter=',')

    # log_file.loc[log_file["name"] == int(name[-7:-3]), ["sog"]] = np.mean(data_nc.variables["sog_radar"])
    # log_file.loc[log_file["name"] == int(name[-7:-3]), ["cog"]] = np.mean(data_nc.variables["cog_radar"])
    # log_file.loc[log_file["name"] == int(name[-7:-3]), ["giro"]] = np.mean(data_nc.variables["giro_radar"])
    # log_file.loc[log_file["name"] == int(name[-7:-3]), ["angle_radon"]] = np.mean(radon_argmax)
    # log_file.loc[log_file["name"] == int(name[-7:-3]), ["angle_std"]] = np.mean(radon_std)
    # #
    # log_file.to_csv("stations_data1.csv", index=False)

    # print("ANGLE_RADON", np.mean(radon_argmax))

    make_anim(back_cartesian_3d_four, full_time, str("pic_" + name[-7:-3]))

    print("station " + name[-7:-3] + " processed and saved")

    data_nc.close()
