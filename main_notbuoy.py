import glob
import netCDF4 as nc
import scipy.signal as ss
import pandas as pd

from src.calculators import *
from src.drawers import *

PATH = '/storage/kubrick/ezhova/WavesOnRadar/'

t_rad = 96  # количество оборотов, учитываемое преобразованием Радона для определения направления
t_four = 256  # количество оборотов, учитываемое преобразованием Фурье
PERIOD_RADAR = 2.5  # период оборота радара
WIDTH_DISPERSION = 11  # ширина в пикселах вырезаемой области Омега из дисперсионного соотношения
cut_ind = 32  # отсечка массива после преобразования Фурье
resolution = 4096  # разрешение картинки по обеим осям (4096 -- максимальное, его лучше не менять)
THRESHOLD = 0.5  # пороговое значение для фильтра к преобразованию Радона
SQUARE_SIZE = 720  # размер вырезаемого квадрата в метрах
SIZE_SQUARE_PIX = int(SQUARE_SIZE / 1.875)  # размер вырезаемого квадрата в пикселах
k_min = 2 * np.pi / SQUARE_SIZE  # минимальное волновое число
k_max = 2 * np.pi / SQUARE_SIZE * cut_ind  # максимальное волновое число

ask = input("manually enter stations or just enter for auto process all stations >> ")

if len(ask) < 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*.nc'))
elif len(ask) == 1:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*.nc'))[:int(ask)]
elif ask[0] == '[':
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*.nc'))[int(ask[1:3]):int(ask[4:6]) + 1]
else:
    stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022-*' + str(f) + '*.nc'))[0] for f in ask.split()]

stations.sort()

an_res = []  # list of obtained directions

mask_circle = make_circle(SIZE_SQUARE_PIX)  # mask of circle for Radon transform

for name in stations:
    data_nc = nc.Dataset(name)  # file data name
    output_file = pd.read_csv(PATH + "sheets/stations_data12.csv", delimiter=",")  # file for output data

    print("station " + name[-17:-6] + " started proccess...")

    # start time in radar data respectfully buoy
    st_ix = 0

    max_time = data_nc.variables["bsktr_radar"].shape[0] - st_ix  # full recorded number of turns

    if max_time < t_four:
        print("!!!!!!!!!!!!!!!!!")
        print(name, "is short, max_time", max_time)
        print("!!!!!!!!!!!!!!!!!")

    if max_time < t_four // 2:  # we can apply mirror not less than half data
        break

    # try cut data from azimuth with maximum dispersion
    back_cart_3d_rad, ang_std = calc_back(data_nc, t_rad, st_ix, max_time, resolution, SQUARE_SIZE, SIZE_SQUARE_PIX, 0,
                                          0, True)

    print("back for radon done")

    print("ang_std >> ", ang_std)

    # make_anim_back(back_cart_3d_rad, "back_ " + name[-17:-6])

    radon_array = radon_process(back_cart_3d_rad, THRESHOLD, mask_circle)
    print("radon done")

    # make_anim_radon(radon_array, "radon_" + name[-17:-6])

    angles, direct = find_main_directions(radon_array, 2, 15)
    print("obtained direction >> ", angles)
    print("obtained directs >> ", direct)

    if np.abs(direct[0]) < 0.1:  # if data so noisy that we can't determine directions
        # try to cut data from 270 azimuth
        back_cart_3d_rad, ang_std = calc_back(data_nc, t_rad, st_ix, max_time, resolution, SQUARE_SIZE, SIZE_SQUARE_PIX,
                                              1, 0, True)
        radon_array = radon_process(back_cart_3d_rad, THRESHOLD, mask_circle)
        angles2, direct2 = find_main_directions(radon_array, 1, 10)
        print("bad angles, new directions >> ", angles2, direct2)
        if np.abs(direct2[0]) > np.abs(direct[0]):
            angles = angles2
            direct = direct2

    direct_std = calc_autocorr(radon_array[:, :, ang_std % 180])
    print("direct std", direct_std)
    if direct_std < 0:
        if ang_std > 180:
            ang_std -= 180
        else:
            ang_std += 180

    # if the direction obtained by Radon is very different from the largest standard deviation,
    # we give priority to the standard deviation
    if np.abs(direct_std) > np.abs(direct[0]) and np.abs(ang_std - angles[0]) > 60:
        angles[0] = ang_std

    print("final", angles[0])

    for an in angles:  # loop in every obtained direction (so we can separate different wave systems)

        # but now we compare only main parameters (because buoy)

        if an == angles[0]:
            back_cartesian_four, ang_std = calc_back(data_nc, t_four, st_ix, max_time, resolution, SQUARE_SIZE, cut_ind,
                                                     2, an, True)
            print("back for fourier done")
            f, res_s = ss.welch(back_cartesian_four, detrend='linear', axis=0, return_onesided=False)
            print("welch done")

            res_s = ss.medfilt(res_s, 5)
            m0, m1, radar_szz = process_fourier(str(PATH + name), res_s, np.median(data_nc.variables["sog_radar"][st_ix: st_ix + t_four]),
                                                an - np.median(data_nc.variables["giro_radar"][st_ix: st_ix + t_four]),
                                                WIDTH_DISPERSION, PERIOD_RADAR, k_min, k_max)

    plt.close()
    plt.plot(np.linspace(0, 1 / PERIOD_RADAR, radar_szz.shape[0]), radar_szz, label="radar")
    plt.legend()
    plt.grid()
    plt.savefig(PATH + "pics/freq_" + str(name[-17:-6]) + ".png")
    plt.show()

    output_file.loc[output_file["name"] == name[-17:-6], ["radar_an"]] = angles[0]

    output_file.loc[output_file["name"] == name[-17:-6], ["radar_m0"]] = m0
    output_file.loc[output_file["name"] == name[-17:-6], ["radar_per"]] = PERIOD_RADAR / (
            np.argmax(radar_szz) / radar_szz.shape[0])
    output_file.loc[output_file["name"] == name[-17:-6], ["radar_an2"]] = angles[-1]

    output_file.to_csv(PATH + "sheets/stations_data12.csv", index=False)

    print("station " + name[-17:-6] + " processed and saved")

    data_nc.close()
