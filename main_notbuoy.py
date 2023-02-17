import time
import glob
import netCDF4 as nc
import scipy.signal as ss
import scipy.fft as sf
import pandas as pd
from skimage.transform import radon
from src.structures import *
from src.calculators import *
from src.drawers import *

ask = input("manually enter stations or just enter for auto process all stations >> ")

if len(ask) < 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*.nc'))
elif len(ask) == 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*.nc'))[:int(ask)]
else:
    stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*' + str(f) + '*.nc'))[0] for f in ask.split()]

stations.sort()

cut_ind = 32
time_radon = 64
time_fourier = 256
PERIOD_RADAR = 2.45
WIDTH_DISPERSION = 5
resolution = 4096
TRASHHOLD = 0.5
SIZE_SQUARE_PIX = 384
k_min = 2 * np.pi / 720
k_max = 2 * np.pi / 720 * cut_ind

an_res = []

mask_circle = make_radon_circle(SIZE_SQUARE_PIX)

for name in stations:
    data_nc = nc.Dataset(name)  # имя файла с данными
    log_file = pd.read_csv("sheets/stations_data12.csv", delimiter=",")

    print("station " + name[-17:-3] + " started proccess...")

    start_ind = 0
    max_time = data_nc.variables["bsktr_radar"].shape[0]

    if max_time < time_fourier:
        print("!!!!!!!!!!!!!!!!!")
        print(name, "is short, max_time", max_time)
        print("!!!!!!!!!!!!!!!!!")

    if max_time < time_fourier // 2:
        break

    back_cartesian_3d_rad = np.ndarray(shape=(time_fourier, SIZE_SQUARE_PIX, SIZE_SQUARE_PIX), dtype=float)
    # back_cartesian_3d_four_one = np.ndarray(shape=(SIZE_SQUARE_PIX, SIZE_SQUARE_PIX), dtype=float)
    radon_array = np.ndarray(shape=(time_radon, SIZE_SQUARE_PIX, 180), dtype=float)
    radon_max = np.ndarray(shape=(time_radon,), dtype=float)

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
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][t + start_ind])
        else:
            back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t - start_ind + 2 * max_time - 1])

        back_cartesian_3d_rad[t] = speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod)
        back_cartesian_3d_rad[t][back_cartesian_3d_rad[t] < 2 * TRASHHOLD * np.mean(back_cartesian_3d_rad[t])] = 0.
        back_cartesian_3d_rad[t][back_cartesian_3d_rad[t] != 0.] = 1.

        radon_array[t] = radon(mask_circle * back_cartesian_3d_rad[t])

    one = True
    angles = find_main_directions(radon_array, 1, 15, 5, one)

    if angles == 666 and one:
        for t in range(time_radon):

            if t % 8 == 0 and t != 0:
                print("radon done " + str(round(t / time_radon * 100, 1)) + "%, estimated time " + str(
                    round((time_radon / t - 1) * (time.time() - time0), 1)))

            area = Area(720, 720, 1360, 180, -180)  # int(data_nc.variables["giro_radar"][t])

            area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
                                                                   data_nc.variables["rad_radar"].shape[0],
                                                                   data_nc.variables["theta_radar"].shape[0],
                                                                   resolution)

            if t < max_time:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][t + start_ind])
            else:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t - start_ind + 2 * max_time - 1])

            back_cartesian_3d_rad[t] = speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod)
            back_cartesian_3d_rad[t][back_cartesian_3d_rad[t] < 2 * TRASHHOLD * np.mean(back_cartesian_3d_rad[t])] = 0.
            back_cartesian_3d_rad[t][back_cartesian_3d_rad[t] != 0.] = 1.

            radon_array[t] = radon(mask_circle * back_cartesian_3d_rad[t])
            one = False
        angles = find_main_directions(radon_array, 1, 15, 5, one)

    print("radon done, obtained directions >> ", angles)

    for an in angles:

        back_cartesian_3d = np.ndarray(shape=(time_fourier, cut_ind, cut_ind), dtype=complex)
        time0 = time.time()

        for t in range(time_fourier):
            if t % 32 == 0 and t != 0:
                print("fourier done " + str(round(t / time_fourier * 100, 1)) + "%, estimated time " + str(
                    round((time_fourier / t - 1) * (time.time() - time0), 1)))

            area = Area(720, 720, 1360, 270, (-270 - an) % 360)  # int(data_nc.variables["giro_radar"][t])

            area_mask_div, area_mask_mod, min_max = make_area_mask(area, data_nc.variables["rad_radar"][-1],
                                                                   data_nc.variables["rad_radar"].shape[0],
                                                                   data_nc.variables["theta_radar"].shape[0],
                                                                   resolution)
            if t < max_time:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][t])
            else:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t + 2 * max_time - 1])

            back_cartesian_3d[t] = sf.fft2(speed_bilinear_interpol(back_polar, area_mask_div, area_mask_mod))[:cut_ind,
                                   :cut_ind]

        f, res_s = ss.welch(back_cartesian_3d, detrend='linear', axis=0, return_onesided=False)

        if an == angles[0]:
            m0, m1, radar_szz = process_fourier(name, res_s,
                                                np.median(data_nc.variables["sog_radar"][
                                                          start_ind: start_ind + time_fourier]),
                                                an + np.median(
                                                    data_nc.variables["giro_radar"][
                                                    start_ind: start_ind + time_fourier]), WIDTH_DISPERSION,
                                                PERIOD_RADAR, k_min, k_max)

    plt.plot(np.linspace(0, 1 / PERIOD_RADAR, radar_szz.shape[0]), radar_szz, label="radar")
    plt.legend()
    plt.grid()
    plt.savefig("pics/freq_" + str(name[-17:-3]) + ".png")
    plt.show()

    log_file.loc[log_file["name"] == name[-17:-3], ["radar_an"]] = angles[0]
    log_file.loc[log_file["name"] == name[-17:-3], ["radar_m0"]] = m0
    log_file.loc[log_file["name"] == name[-17:-3], ["radar_per"]] = PERIOD_RADAR / (
            np.argmax(radar_szz) / radar_szz.shape[0])

    if len(angles) > 1:
        log_file.loc[log_file["name"] == name[-17:-3], ["radar_an2"]] = angles[1]

    log_file.to_csv("sheets/stations_data12.csv", index=False)

    print("station " + name[-17:-3] + " processed and saved")
    data_nc.close()