import glob
import netCDF4 as nc
import scipy.signal as ss
import pandas as pd
from src.back import Back
from src.calculators import *
from src.drawers import *

PATH = '/storage/kubrick/ezhova/WavesOnRadar/'

t_rad = 32  # количество оборотов, учитываемое преобразованием Радона для определения направления
t_four = 256  # количество оборотов, учитываемое преобразованием Фурье
PERIOD_RADAR = 2.5  # период оборота радара
WIDTH_DISPERSION = 15  # ширина в пикселах вырезаемой области Омега из дисперсионного соотношения
cut_ind = 32  # отсечка массива после преобразования Фурье
resolution = 4096  # разрешение картинки по обеим осям (4096 -- максимальное, его лучше не менять)
THRESHOLD = 0.5  # пороговое значение для фильтра к преобразованию Радона
SQUARE_SIZE = 720  # размер вырезаемого квадрата в метрах
SIZE_SQUARE_PIX = int(SQUARE_SIZE / 1.875)  # размер вырезаемого квадрата в пикселах
k_min = 2 * np.pi / SQUARE_SIZE  # минимальное волновое число
k_max = 2 * np.pi / SQUARE_SIZE * cut_ind  # максимальное волновое число

exp = input("enter number of expedition >> ")

if len(exp) == 2:
    ask = input("manually enter stations or just enter for auto process all stations >> ")

    if len(ask) < 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI' + exp + '/nc/20*.nc'))
    elif len(ask) == 1:
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI' + exp + '/nc/20*.nc'))[:int(ask)]
    elif ask[0] == '[':
        stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI' + exp + '/nc/20*.nc'))[int(ask[1:3]):int(ask[4:6]) + 1]
    else:
        stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI' + exp + '/nc/20*' + str(f) + '*.nc'))[0] for f in ask.split()]
else:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI58/nc/20*.nc'))
    stations2 = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/20*.nc'))
    stations3 = glob.glob(str('/storage/tartar/DATA/RADAR/AI57/nc/20*.nc'))
    stations += stations2
    stations += stations3

stations.sort()

for st in stations:
    tmp = nc.Dataset(st)
    print(st[-20:-6], tmp.variables["bsktr_radar"].shape[0])

an_res = []  # list of obtained directions

mask_circle = make_circle(SIZE_SQUARE_PIX)  # mask of circle for Radon transform

for name in stations:
    data_nc = nc.Dataset(name)  # file data name

    output_file = pd.read_csv(PATH + "sheets/stations_data12.csv", delimiter=",")  # file for output data

    print("station " + name[-20:-6] + " started proccess...")

    # start time in radar data respectfully buoy
    st_ix = 0
    max_ix = data_nc.variables["bsktr_radar"].shape[0] - st_ix  # full recorded number of turns

    if max_ix < t_four / 2:
        print('ATTENTION ATTENTION ATTENTION', name[-20:-6])

    if max_ix > t_four / 2:  # we can apply mirror not less than half data

        back = Back(data_nc, t_rad, t_four, st_ix, max_ix, mask_circle, THRESHOLD)
        back.calc_radon(0)

        angles = back.directions_search(1, 15)

        m0, m1, radar_szz, angle_aa = back.calc_fourier(angles, 32, WIDTH_DISPERSION, PERIOD_RADAR, name[-7:-3])

        # plt.close()
        # plt.plot(np.linspace(0, 1 / PERIOD_RADAR, radar_szz.shape[0]), radar_szz, label="radar")
        # plt.legend()
        # plt.grid()
        # plt.savefig(PATH + "pics/freq_" + str(name[-17:-6]) + ".png")
        # plt.show()

        output_file.loc[output_file["name"] == name[-20:-6], ["radar_an"]] = angles[0]

        output_file.loc[output_file["name"] == name[-20:-6], ["radar_m0"]] = m0
        output_file.loc[output_file["name"] == name[-20:-6], ["radar_per"]] = PERIOD_RADAR / (
                np.argmax(radar_szz) / radar_szz.shape[0])
        output_file.loc[output_file["name"] == name[-20:-6], ["radar_an2"]] = angles[-1]
        output_file.loc[output_file["name"] == name[-20:-6], ["radar_an3"]] = angle_aa
        output_file.loc[output_file["name"] == name[-20:-6], ["lat_radar"]] = np.mean(
            data_nc.variables["lat_radar"][st_ix: st_ix + t_four])
        output_file.loc[output_file["name"] == name[-20:-6], ["lon_radar"]] = np.mean(
            data_nc.variables["lon_radar"][st_ix: st_ix + t_four])
        output_file.loc[output_file["name"] == name[-20:-6], ["speed"]] = np.median(
            data_nc.variables["sog_radar"][st_ix: st_ix + t_four])
        output_file.loc[output_file["name"] == name[-20:-6], ["hdg"]] = np.median(
            data_nc.variables["giro_radar"][st_ix: st_ix + t_four])
        output_file.loc[output_file["name"] == name[-20:-6], ["dir_std"]] = back.ang_std

        output_file.to_csv(PATH + "sheets/stations_data12.csv", index=False)

        print("station " + name[-20:-6] + " processed and saved")

    data_nc.close()
