import time

import numpy as np
from scipy.signal import welch
from src.calculators import bilinear_interpol, find_main_directions, calc_forward_toward, calc_dispersion
from src.area import *
import scipy.fft as sf
from skimage.transform import radon
import matplotlib.pyplot as plt

CUT_IND = 32


class Back:
    """
    класс
    """

    def __init__(self, data, radon_time: int, fourier_time: int, start_index: int,
                 max_index: int, circle_radon: np.ndarray, threshold=0.4, resolution=4096, size_square=720,
                 size_square_px=384):
        """
        конструктор
        """
        self.data = data
        self.radon_time = radon_time
        self.fourier_time = fourier_time
        self.circle_radon = circle_radon
        self.resolution = resolution
        self.size_square = size_square
        self.size_square_px = size_square_px
        self.speed = float(np.median(data.variables["sog_radar"][start_index: start_index + fourier_time]))
        self.start_index = start_index
        self.threshold = threshold
        self.radon_array = np.zeros(shape=(radon_time, size_square_px, 180), dtype=np.double)
        self.dir_std = 0.

        # full recorded number of turns
        self.max_index = max_index

    def calc_std(self, start_ind: int = 0, max_time: int = 512):
        """
        calculating dispersion of some serial shots to find more clean zone of data
        @param data_bsktr: input netcdf data
        @param start_ind: zero buoy time shot
        @param max_time: full number of recorded shots
        @return azimuth and distance of clear zone
        """

        bck_cntr = np.ndarray(shape=(4, self.resolution, int(self.resolution // 8)), dtype=np.int)
        for t in range(bck_cntr.shape[0]):

            if t < max_time - 1:
                bck_cntr[t] = self.data.variables["bsktr_radar"][t + start_ind][:,
                              self.resolution // 8:self.resolution // 4]
            else:
                bck_cntr[t] = self.data.variables["bsktr_radar"][- t - start_ind + 2 * max_time - 1][:,
                              self.resolution // 8:self.resolution // 4]

        std_array = np.std(bck_cntr, axis=0)

        max_std = np.max(std_array)
        mean_std =np.mean(std_array)

        std_array_smooth = np.zeros(shape=(int(std_array.shape[0] // 64), int(std_array.shape[1] // 128)))

        for i in range(std_array_smooth.shape[0]):
            for j in range(std_array_smooth.shape[1]):
                std_array_smooth[i, j] = np.mean(std_array[i * 64: i * 64 + 63, j * 128: j * 128 + 127])

        argmax = np.argmax(std_array_smooth)
        theta = (argmax // std_array_smooth.shape[1]) / std_array_smooth.shape[0] * 360
        rho = (argmax % std_array_smooth.shape[1] + 512) * 1.875

        if np.abs(theta - 6) < 6:  # 6 degrees is gluing place, we need avoid it
            theta += np.sign(theta - 6) * 2 * np.abs(theta - 6)

        self.dir_std = round(theta)

        return theta, rho, max_std, mean_std

    def calc_back(self, radon_or_fourier: bool, an: float = 0.):
        """
        calculating a cartesian data inside current area
        @param radon_or_fourier: True for Radon, False for Fourier
        @param an: obtained by radon angle
        @return cartesian data inside area
        """
        if radon_or_fourier:
            back = np.zeros(shape=(self.radon_time, self.size_square_px, self.size_square_px), dtype=np.double)
        else:
            back = np.zeros(shape=(self.fourier_time, CUT_IND, CUT_IND), dtype=np.complex_)

        for t in range(back.shape[0]):

            dir_std, rad_std, _, __ = self.calc_std()

            if radon_or_fourier:
                zone = Area(self.size_square, self.size_square, rad_std, dir_std, 0)
            else:
                zone = Area(self.size_square, self.size_square, rad_std, dir_std, (-dir_std - an) % 360)

            area_mask_div, area_mask_mod, min_max = make_area_mask(zone, self.data.variables["rad_radar"][-1],
                                                                   self.data.variables["rad_radar"].shape[0],
                                                                   self.data.variables["theta_radar"].shape[0],
                                                                   self.resolution)
            if t < self.max_index:
                back_polar = np.transpose(self.data.variables["bsktr_radar"][t + self.start_index])
            else:
                back_polar = np.transpose(
                    self.data.variables["bsktr_radar"][- t - self.start_index + 2 * self.max_index - 1])

            if radon_or_fourier:
                t0 = time.time()
                back[t] = bilinear_interpol(back_polar, area_mask_div, area_mask_mod)
                print(time.time() - t0)

            else:
                back[t] = sf.fft2(bilinear_interpol(back_polar, area_mask_div, area_mask_mod))[:CUT_IND, :CUT_IND]

        return back

    def calc_radon(self):
        """

        """
        # try cut data from azimuth with maximum dispersion
        back_cart_3d_rad = self.calc_back(True)
        print("back for radon done")
        print("dir_std >> ", self.dir_std)

        self.radon_array = np.zeros(shape=(self.radon_time, self.size_square_px, 180), dtype=np.double)
        for t in range(self.radon_time):
            # filtering
            back_cart_3d_rad[t][back_cart_3d_rad[t] < 2 * self.threshold * np.mean(back_cart_3d_rad[t])] = 0
            back_cart_3d_rad[t][back_cart_3d_rad[t] != 0] = 1
            # radon transform
            self.radon_array[t] = radon(self.circle_radon * back_cart_3d_rad[t])

        return self.radon_array

    def directions_search(self, peak_numbers: int, peak_window: int) -> list:

        angles, direct = find_main_directions(self.radon_array, peak_numbers, peak_window)
        print("obtained angles >> ", angles)
        print("obtained directs >> ", direct)

        direct_std = calc_forward_toward(self.radon_array[:, :, self.dir_std % 180], 25)
        print("direct std", direct_std)
        if direct_std < 0:
            if self.dir_std > 180:
                self.dir_std -= 180
            else:
                self.dir_std += 180

        # if the direction obtained by Radon is so different from the largest standard deviation,
        # we give priority to the standard deviation
        if np.abs(direct_std) > np.abs(direct[0]) and np.abs(self.dir_std - angles[0]) > 60:
            angles[0] = self.dir_std

        print("final", angles[0])
        return angles

    def calc_fourier(self, angle, cut_ind: int, disp_width: int, turn_period: float, name):

        print("back for fourier start")
        back_cartesian_four = self.calc_back(False, angle)
        print("back for fourier done")
        f, res_s = welch(back_cartesian_four, detrend='linear', axis=0, return_onesided=False)
        print("welch done")

        k_max = 2 * np.pi / self.size_square * cut_ind

        return calc_dispersion(name, res_s, self.speed, disp_width, turn_period, k_max)
