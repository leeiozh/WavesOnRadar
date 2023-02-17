import glob
import netCDF4 as nc
import scipy.signal as ss
import pandas as pd

from src.calculators import *
from src.drawers import *

ask = input("manually enter stations or just enter for auto process all stations >> ")

if len(ask) < 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))
elif len(ask) == 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*.nc'))[:int(ask)]
else:
    stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/0606*' + str(f) + '*.nc'))[0] for f in ask.split()]

stations.sort()

t_rad = 72  # количество оборотов, учитываемое преобразованием Радона для определения направления
t_four = 720  # количество оборотов, учитываемое преобразованием Фурье
PERIOD_RADAR = 2.5  # период оборота радара
WIDTH_DISPERSION = 7  # ширина в пикселах вырезаемой области Омега из дисперсионного соотношения
cut_ind = 32  # отсечка массива после преобразования Фурье
resolution = 4096  # разрешение картинки по обеим осям (4096 -- максимальное, его лучше не менять)
THRESHOLD = 0.5  # пороговое значение для фильтра к преобразованию Радона
SQUARE_SIZE = 720  # размер вырезаемого квадрата в метрах
SIZE_SQUARE_PIX = SQUARE_SIZE / 1.875  # размер вырезаемого квадрата в пикселах
k_min = 2 * np.pi / SQUARE_SIZE  # минимальное волновое число
k_max = 2 * np.pi / SQUARE_SIZE * cut_ind  # максимальное волновое число

an_res = []
r = []

mask_circle = make_radon_circle(SIZE_SQUARE_PIX)

for name in stations:
    data_nc = nc.Dataset(name)  # имя файла с данными
    log_file = pd.read_csv("sheets/stations_data13.csv", delimiter=",")

    print("station " + name[-7:-3] + " started proccess...")

    st_ix = np.argmax(np.array(data_nc.variables["time_radar"]) - data_nc.variables["time_buoy"][0] >= 0)

    max_time = data_nc.variables["bsktr_radar"].shape[0] - st_ix

    if max_time < t_four:
        print("!!!!!!!!!!!!!!!!!")
        print(name, "is short, max_time", max_time)
        print("!!!!!!!!!!!!!!!!!")

    if max_time < t_four // 2:
        break

    back_cart_3d_rad = calc_back(data_nc, t_rad, st_ix, max_time, resolution, SQUARE_SIZE, SIZE_SQUARE_PIX, 0, 0, True)
    print("back for radon done")

    # make_anim_back(back_cart_3d_rad, "back_ " + name[-7:-3])

    radon_array = thresh_radon(back_cart_3d_rad, SIZE_SQUARE_PIX, THRESHOLD, mask_circle)
    print("radon done")

    # make_anim_radon(radon_array, "radon_" + name[-7:-3])

    one = True
    angles = find_main_directions(radon_array, 1, 15, 5, one)
    print("obtained direction >> ", angles)

    if angles == 666 and one:
        back_cart_3d_rad = calc_back(data_nc, t_rad, st_ix, max_time, resolution, SQUARE_SIZE, SIZE_SQUARE_PIX, 1, 0, True)
        radon_array = thresh_radon(back_cart_3d_rad, SIZE_SQUARE_PIX, THRESHOLD, mask_circle)
        one = False
        angles = find_main_directions(radon_array, 1, 15, 5, one)
        print("bad angles, new directions >> ", angles)

    # angles = log_file.loc[log_file["name"] == int(name[-7:-3]), ["radar_an"]].to_numpy()[0]

    for an in angles:

        back_cartesian_four = calc_back(data_nc, t_four, st_ix, max_time, resolution, SQUARE_SIZE, cut_ind, 2, an, True)
        print("back for fourier done")
        f, res_s = ss.welch(back_cartesian_four, detrend='linear', axis=0, return_onesided=False)
        print("welch done")

        # res_s = ss.medfilt(res_s, 5)

        if an == angles[0]:
            m0, m1, radar_szz = process_fourier(res_s, np.median(data_nc.variables["sog_radar"][st_ix: st_ix + t_four]),
                                                an + np.median(data_nc.variables["giro_radar"][st_ix: st_ix + t_four]),
                                                WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max)

    buoy_freq = data_nc.variables["freq_manual"]
    buoy_Szz = data_nc.variables["Szz_manual"]
    buoy_swh = data_nc.variables["H0_manual"]
    buoy_per = data_nc.variables["Tp_manual"]
    buoy_ang = data_nc.variables["Theta_p_manual"]

    plt.close()
    plt.plot(np.linspace(0, 1 / PERIOD_RADAR, radar_szz.shape[0]),
             radar_szz / np.max(radar_szz) * np.max(np.array(buoy_Szz[:40])), label="radar")
    plt.plot(buoy_freq[:40], buoy_Szz[:40], label="buoy")
    plt.legend()
    plt.grid()
    plt.savefig("pics/freq_" + str(name[-7:-3]) + ".png")
    plt.show()

    log_file.loc[log_file["name"] == int(name[-7:-3]), ["radar_an"]] = angles[0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["radar_m0"]] = m0
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["radar_per"]] = PERIOD_RADAR / (
            np.argmax(radar_szz) / radar_szz.shape[0])
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["buoy_swh"]] = buoy_swh[0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["buoy_per"]] = buoy_per[0]
    log_file.loc[log_file["name"] == int(name[-7:-3]), ["buoy_ang"]] = buoy_ang[0]

    if len(angles) > 1:
        log_file.loc[log_file["name"] == int(name[-7:-3]), ["radar_an2"]] = angles[1]

    log_file.to_csv("sheets/stations_data13.csv", index=False)

    print("station " + name[-7:-3] + " processed and saved")

    data_nc.close()
