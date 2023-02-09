import math

import numpy as np
from numba import njit, int32, float64
from numba.experimental import jitclass

area_spec = [('width', float64),
             ('height', float64),
             ('distance', float64),
             ('azimuth', float64),
             ('orientation', float64)]


@jitclass(area_spec, )
class Area:
    """
    класс исследуемого прямоугольника
    """

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
        res = np.zeros((4, 2))
        res[3] = center + lu_rd_tmp
        res[0] = center + ru_ld_tmp
        res[2] = center - ru_ld_tmp
        res[1] = center - lu_rd_tmp

        return res

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


@njit
def make_area_mask(area: Area, rad_limit: int, rad_mesh_shape: int, theta_mesh_shape: int, resolution: int):
    """
    создание "матрицы" перехода от полярных к декартовым для указанной области (вычисляется один раз)
    """
    min_max = np.array([4096, 0])

    res_x_size = math.floor(area.width / rad_limit * resolution)  # размер массива результатов по х
    res_y_size = math.floor(area.height / rad_limit * resolution)  # размер массива результатов по у

    coords_vertex = area.get_coords_vertex()
    coords_vertex /= rad_limit
    coords_vertex *= rad_mesh_shape  # координаты вершин области в единицах исходного массива

    res_mask_div = np.zeros(shape=(2, res_x_size, res_y_size), dtype=int32)
    res_mask_mod = np.zeros(shape=(2, res_x_size, res_y_size), dtype=float64)

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
