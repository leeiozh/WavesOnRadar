import glob
import numpy as np

import pandas as pd
import netCDF4 as nc
from src.calculators import *
from src.drawers import *
from src.back import Back
from numba import float64

PATH = '/storage/kubrick/ezhova/WavesOnRadar/'

t_rad = 32  # количество оборотов, учитываемое преобразованием Радона для определения направления
t_four = 512  # количество оборотов, учитываемое преобразованием Фурье
PERIOD_RADAR = 2.5  # период оборота радара
WIDTH_DISPERSION = 15  # (полуширина + 1) в пикселах вырезаемой области Омега из дисперсионного соотношения
cut_ind = 32  # отсечка массива после преобразования Фурье
resolution = 4096  # разрешение картинки по обеим осям (4096 -- максимальное, его лучше не менять)
THRESHOLD = 0.5  # пороговое значение для фильтра к преобразованию Радона
SQUARE_SIZE = 720  # размер вырезаемого квадрата в метрах
SIZE_SQUARE_PIX = int(SQUARE_SIZE / 1.875)  # размер вырезаемого квадрата в пикселах
k_min = 2 * np.pi / SQUARE_SIZE  # минимальное волновое число
k_max = 2 * np.pi / SQUARE_SIZE * cut_ind  # максимальное волновое число

exp = input("enter number of expedition >> ")

if exp == '63':

    ask = input("manually enter stations or just enter for auto process all stations >> ")

    if len(ask) < 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))
    elif len(ask) == 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))[:int(ask)]
    elif ask[0] == '[':
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))[int(ask[1:3]):int(ask[4:6]) + 1]
    else:
        stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*' + str(f) + '*.nc'))[0] for f in
                    ask.split()]

elif exp == '58':
    ask = input("manually enter stations or just enter for auto process all stations >> ")

    if len(ask) < 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI58/nc/0606*.nc'))
    elif len(ask) == 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI58/nc/0606*.nc'))[:int(ask)]
    elif ask[0] == '[':
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI58/nc/0606*.nc'))[int(ask[1:3]):int(ask[4:6]) + 1]
    else:
        stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI58/nc/0606*' + str(f) + '*.nc'))[0] for f in
                    ask.split()]

elif exp == '57':
    ask = input("manually enter stations or just enter for auto process all stations >> ")

    if len(ask) < 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI57/nc/0606*.nc'))
    elif len(ask) == 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI57/nc/0606*.nc'))[:int(ask)]
    elif ask[0] == '[':
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI57/nc/0606*.nc'))[int(ask[1:3]):int(ask[4:6]) + 1]
    else:
        stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI57/nc/0606*' + str(f) + '*.nc'))[0] for f in
                    ask.split()]

else:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI58/nc/0606*.nc'))
    stations2 = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))
    stations3 = glob.glob(str('/storage/tartar/DATA/RADAR/AI57/nc/0606*.nc'))
    stations += stations2
    stations += stations3

stations.sort()

an_res = []  # list of obtained directions

mask_circle = make_circle(SIZE_SQUARE_PIX)  # mask of circle for Radon transform

for name in stations:
    data_nc = nc.Dataset(name)  # file data name
    output_file = pd.read_csv(PATH + "sheets/stations_data13.csv", delimiter=",")  # file for output data

    print("station " + name[-7:-3] + " started proccess...")

    st_ix = np.argmax(np.array(data_nc.variables["time_radar"]) - data_nc.variables["time_buoy"][0] >= 0)
    max_ix = data_nc.variables["bsktr_radar"].shape[0] - st_ix

    if max_ix < t_four // 2:  # we can apply mirror not less than half data
        print(name[-7:-3], '!!!! number of turns is not enough for fourier process =(, max_index =>', max_ix,
                         'fourier_time =>', t_four)
        break

    back = Back(data_nc, t_rad, t_four, st_ix, max_ix, mask_circle, THRESHOLD)
    print(back.calc_std(st_ix, max_ix))
    back.calc_radon(0)

    # for i in range(10):
    #    make_shot(radon_array[i], "radon_" + name[-7:-3] + str(i), True)

    angles = back.directions_search(1, 15)

    m0, m1, radar_szz, angle_aa = back.calc_fourier(angles, 32, WIDTH_DISPERSION, PERIOD_RADAR, name)

    # read parameters obtained by buoy for comparing
    buoy_freq = data_nc.variables["freq_manual"]
    buoy_Szz = data_nc.variables["Szz_manual"]
    buoy_swh = data_nc.variables["H0_manual"]
    buoy_per = data_nc.variables["Tp_manual"]
    buoy_ang = data_nc.variables["Theta_p_manual"]

    # plt.close()
    # plt.plot(np.linspace(0, 1 / PERIOD_RADAR, radar_szz.shape[0]),
    #          radar_szz / np.max(radar_szz) * np.max(np.array(buoy_Szz[:40])), label="radar")
    # plt.plot(buoy_freq[:40], buoy_Szz[:40], label="buoy")
    # plt.legend()
    # plt.grid()
    # plt.savefig("pics/freq_" + str(name[-7:-3]) + ".png")
    # plt.show()

    output_file.loc[output_file["name"] == int(name[-7:-3]), ["radar_an"]] = angles[0]

    output_file.loc[output_file["name"] == int(name[-7:-3]), ["radar_m0"]] = m0
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["radar_per"]] = PERIOD_RADAR / (
            np.argmax(radar_szz) / radar_szz.shape[0])
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["buoy_swh"]] = buoy_swh[0]
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["buoy_per"]] = buoy_per[0]
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["buoy_ang"]] = buoy_ang[0]

    output_file.loc[output_file["name"] == int(name[-7:-3]), ["radar_an2"]] = angles[-1]
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["radar_an3"]] = angle_aa
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["wdir"]] = np.median(
        data_nc.variables["wdir"][st_ix: st_ix + t_four])
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["speed"]] = np.median(
        data_nc.variables["sog_radar"][st_ix: st_ix + t_four])
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["hdg"]] = np.median(
        data_nc.variables["giro_radar"][st_ix: st_ix + t_four])
    output_file.loc[output_file["name"] == int(name[-7:-3]), ["dir_std"]] = back.ang_std

    output_file.to_csv(PATH + "sheets/stations_data13.csv", index=False)

    print("station " + name[-7:-3] + " processed and saved")

    data_nc.close()
