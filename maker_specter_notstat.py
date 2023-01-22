import time
import glob
import netCDF4 as nc
import scipy.signal as ss
import scipy.fft as sf
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from skimage.transform import radon


class Area:
    """
    класс исследуемого прямоугольника
    """
    width: float  # ширина прямоугольника м
    height: float  # высота прямоугольника м
    distance: float  # расстояние от радара до центра прямоугольника м
    azimuth: float  # угол между осью у и направлением на центр прямоугольника (по часовой) рад
    orientation: float  # угол между осью у и направлением на середину верхнего ребра прямоугольника (по часовой) рад

    def __init__(self, width: float, height: float, distance: float, azimuth: float, orientation: float):
        """
        конструктор
        :param width: ширина прямоугольника (м)
        :param height: высота прямоугольника (м)
        :param distance:  расстояние от радара до центра прямоугольника (м)
        :param azimuth: угол между осью у и направлением на центр прямоугольника (по часовой) (град)
        :param orientation: угол между осью у и направлением на середину верхнего ребра прямоугольника (по часовой) (град)
        """
        self.width = width
        self.height = height
        self.distance = distance
        self.azimuth = (azimuth / 180) * np.pi
        self.orientation = orientation / 180 * np.pi

    def get_diag_params(self):
        """
        геттер параметров диагонали
        :return: длина диагонали, угол между направлением на центр верхней грани и правую верхнюю вершину
        """
        return np.sqrt(self.width ** 2 + self.height ** 2), np.arctan2(self.width, self.height)

    def get_center(self):
        """
        геттер положения центра
        :return: [x_center, y_center] (м)
        """
        return np.array([self.distance * np.cos(self.azimuth), self.distance * np.sin(self.azimuth)])

    def get_coords_vertex(self):
        """
        геттер координат вершин прямоугольника в декартовой системе
        :return: [правый верхний, правый нижний, левый нижний, левый верхний] м
        """
        diag, beta = self.get_diag_params()
        center = self.get_center()
        angle_min = beta - self.orientation
        angle_plus = beta + self.orientation
        lu_rd_tmp = 0.5 * diag * np.array([-np.cos(angle_plus), np.sin(angle_plus)])
        ru_ld_tmp = 0.5 * diag * np.array([np.cos(angle_min), np.sin(angle_min)])
        lu = center + lu_rd_tmp
        ru = center + ru_ld_tmp
        ld = center - ru_ld_tmp
        rd = center - lu_rd_tmp
        return np.array([ru, rd, ld, lu])

    def get_coords_vertex_map(self):
        """
        геттер координат вершин прямоугольника в декартовой системе, сохраняя ориентацию исходника
        :return: [правый верхний, правый нижний, левый нижний, левый верхний] м
        """
        diag, beta = self.get_diag_params()
        center = self.get_center()
        tmp_x = 0.5 * diag * np.cos(beta - self.orientation)
        tmp_y = 0.5 * diag * np.sin(beta + self.orientation)
        return np.array([center[1] - tmp_y, center[1] + tmp_y, center[0] - tmp_x, center[0] + tmp_x])

    def get_coords_rect(self):
        angle = np.pi / 2 - self.azimuth
        center = self.distance * np.array([np.cos(angle), np.sin(angle)])
        return center - np.array(self.width * np.cos(self.orientation), self.height * np.sin(self.orientation)) / 2


def make_area_mask(area: Area, rad_limit, rad_mesh_shape, theta_mesh_shape, resolution):
    min_max = np.array([4096, 0])

    r_lim = rad_limit
    w = area.width
    h = area.height  # в метрах
    w = min(w, r_lim)
    h = min(h, r_lim)
    res_x_size = round(w / r_lim * resolution)  # размер массива результатов по х
    res_y_size = round(h / r_lim * resolution)  # размер массива результатов по у

    coords_vertex = area.get_coords_vertex()
    coords_vertex /= rad_limit
    coords_vertex *= rad_mesh_shape  # координаты вершин области в единицах исходного массива

    res_mask_div = np.ndarray(shape=(2, res_x_size, res_y_size), dtype=int)
    res_mask_mod = np.ndarray(shape=(2, res_x_size, res_y_size), dtype=float)

    for i in range(res_x_size):
        x = np.linspace(coords_vertex[2, 0] + i / res_x_size * (coords_vertex[1, 0] - coords_vertex[2, 0]),
                        coords_vertex[3, 0] + i / res_x_size * (coords_vertex[0, 0] - coords_vertex[3, 0]),
                        res_y_size)
        y = np.linspace(coords_vertex[2, 1] + i / res_x_size * (coords_vertex[1, 1] - coords_vertex[2, 1]),
                        coords_vertex[3, 1] + i / res_x_size * (coords_vertex[0, 1] - coords_vertex[3, 1]),
                        res_y_size)

        r = np.sqrt(x ** 2 + y ** 2)
        r[r >= 4095] = 0

        th = -np.arctan2(x, y)
        th /= (2 * np.pi)
        th += 0.25
        th *= theta_mesh_shape

        res_mask_div[0, i, :] = np.floor(r).astype('int')
        res_mask_div[1, i, :] = np.floor(th).astype('int')
        res_mask_mod[0, i, :] = r - res_mask_div[0, i, :]
        res_mask_mod[1, i, :] = th - res_mask_div[1, i, :]

    min_max[0] = min(min_max[0], np.min(res_mask_div[0]))
    min_max[1] = max(min_max[1], np.max(res_mask_div[0]))

    return res_mask_div, res_mask_mod, min_max


def make_abs_k_mask(data_shape_1):
    mask_k = np.zeros((data_shape_1, data_shape_1, data_shape_1))
    for i in range(data_shape_1 - 1):
        mask_k[i, 0, 0] = 1
        for j in range(i):
            val = np.sqrt(i * i - j * j)
            div = int(np.sqrt(i * i - j * j))
            mod = val - div
            mask_k[i, j, div] = 1 - mod
            mask_k[i, div, j] = 1 - mod
            mask_k[i, j, div + 1] = mod
            mask_k[i, div + 1, j] = mod
    return mask_k


def spec_3dto2d(mask, data_3d):
    res = np.zeros((data_3d.shape[0], data_3d.shape[1]))

    for t in range(data_3d.shape[0]):
        for k in range(data_3d.shape[1]):
            res[t, k] = np.mean(mask[k] * data_3d[t])

    return res


ask = int(input("press 0 for manual input stations or 1 for auto process all stations >> "))

if ask != 0:
    stations = glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022*.nc'))
else:
    stations = [glob.glob(str('/storage/tartar/DATA/RADAR/AI63/nc/2022*' + str(f) + '*.nc'))[0] for f in
                input("enter stations >> ").split()]

stations.sort()

cut_ind = 64
full_time = 256
PERIOD_RADAR = 2.5
resolution = 4096

mask_circle = np.zeros((384, 384))
for x in range(384):
    for y in range(384):
        if ((x - 192) * (x - 192) + (y - 192) * (y - 192)) < 192 * 192:
            mask_circle[x, y] = 1.

log_file = pd.read_csv("notstations_data.csv", delimiter=",")
j = -1

for name in stations:
    j += 1
    if j >= 28:
        log_file = pd.read_csv("notstations_data.csv", delimiter=",")
        data_nc = nc.Dataset(name)  # имя файла с данными

        print("station " + name[-17:-6] + " started proccess...")
        print("***************")
        print("GIRO_RADAR =", np.mean(data_nc.variables["giro_radar"]))
        print("SOG_RADAR =", np.mean(data_nc.variables["sog_radar"]))
        print("COG_RADAR =", np.mean(data_nc.variables["cog_radar"]))
        print("***************")
        time0 = time.time()

        max_time = data_nc.variables["bsktr_radar"].shape[0]

        if max_time < full_time:
            print("!!!!!!!!!!!!!!!!!")
            print(name, "is short, max_time", max_time)
            print("!!!!!!!!!!!!!!!!!")

        if max_time < full_time // 2:
            break

        area_arr = Area(720, 720, 1360, data_nc.variables["giro_radar"][0] + 45, 0)

        area_mask_div, area_mask_mod, min_max = make_area_mask(area_arr, data_nc.variables["rad_radar"][-1],
                                                               data_nc.variables["rad_radar"].shape[0],
                                                               data_nc.variables["theta_radar"].shape[0], resolution)

        back_cartesian_3d_four = np.ndarray(shape=(full_time, cut_ind, cut_ind), dtype=complex)
        radon_argmax = np.ndarray(shape=(full_time,), dtype=float)

        for t in range(full_time):

            if t % 13 == 0 and t != 0:
                print("done " + str(round(t / full_time * 100, 1)) + "%, estimated time " + str(
                    round((full_time / t - 1) * (time.time() - time0) / 60, 1)))

            if t < max_time:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][t])
            else:
                back_polar = np.transpose(data_nc.variables["bsktr_radar"][- t + 2 * max_time - 1])

            back_cartesian_3d_0 = np.zeros((area_mask_div.shape[1], area_mask_div.shape[2]))

            for i in range(area_mask_div.shape[1]):
                for j in range(area_mask_div.shape[2]):
                    back_cartesian_3d_0[i, j] = (1 - area_mask_mod[0, i, j]) * (1 - area_mask_mod[1, i, j]) * \
                                                back_polar[area_mask_div[0, i, j], area_mask_div[1, i, j]] + \
                                                (1 - area_mask_mod[0, i, j]) * area_mask_mod[1, i, j] * \
                                                back_polar[area_mask_div[0, i, j], area_mask_div[1, i, j] + 1] + \
                                                area_mask_mod[0, i, j] * (1 - area_mask_mod[1, i, j]) * \
                                                back_polar[area_mask_div[0, i, j] + 1, area_mask_div[1, i, j]] + \
                                                area_mask_mod[0, i, j] * area_mask_mod[1, i, j] * back_polar[
                                                    area_mask_div[0, i, j] + 1, area_mask_div[1, i, j] + 1]

            radon_argmax[t] = np.argmax(np.max(radon(mask_circle * back_cartesian_3d_0), axis=0))

            # plt.imshow(radon(mask_circle * back_cartesian_3d_0), origin='lower', cmap='gnuplot', aspect=0.5)
            # plt.savefig("radon" + str(t) + ".png")

            print(radon_argmax[t], end=' ')
            back_cartesian_3d_four[t] = sf.fft2(back_cartesian_3d_0)[:cut_ind, :cut_ind]

            # back_cartesian_3d_four[t][cut_ind:, cut_ind:] = sf.fft2(back_cartesian_3d_0)[:cut_ind, :cut_ind]
            # back_cartesian_3d_four[t][cut_ind:, :cut_ind] = np.rot90(sf.fft2(np.flipud(back_cartesian_3d_0))[:cut_ind, :cut_ind], k=1)
            # back_cartesian_3d_four[t][:cut_ind, cut_ind:] = np.rot90(sf.fft2(np.fliplr(back_cartesian_3d_0))[:cut_ind, :cut_ind], k=3)
            # back_cartesian_3d_four[t][:cut_ind, :cut_ind] = np.rot90(sf.fft2(np.flipud(np.fliplr(back_cartesian_3d_0)))[:cut_ind, :cut_ind], k=2)

        freq, res_s = ss.welch(back_cartesian_3d_four, detrend='linear', axis=0, return_onesided=False)

        print("fourier done, start saving ...")

        res_resh = res_s.reshape(res_s.shape[0], -1)
        np.savetxt('results/specter_' + name[-17:-6] + '.csv', res_resh, delimiter=',')

        logs = np.array([np.mean(data_nc.variables["sog_radar"]), np.mean(data_nc.variables["cog_radar"]),
                         np.mean(data_nc.variables["giro_radar"]), np.mean(radon_argmax)])

        log_file.at[j, "name"] = name[-17:-6]
        log_file.at[j, "sog"] = np.mean(data_nc.variables["sog_radar"])
        log_file.at[j, "cog"] = np.mean(data_nc.variables["cog_radar"])
        log_file.at[j, "giro"] = np.mean(data_nc.variables["giro_radar"])
        log_file.at[j, "angle_radon"] = np.mean(radon_argmax)

        log_file.to_csv("notstations_data.csv", index=False)

        print("ANGLE_RADON", np.mean(radon_argmax))

        print("station " + name[-17:-6] + " processed and saved")

        data_nc.close()

