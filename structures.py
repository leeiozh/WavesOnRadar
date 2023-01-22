import numpy as np


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


def make_area_mask(area, rad_limit, rad_mesh_shape, theta_mesh_shape, resolution):
    """
    создание "матрицы" перехода от полярных к декартовым для указанной области (вычисляется один раз)
    """
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
