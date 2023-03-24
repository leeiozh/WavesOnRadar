import numpy as np
from scipy.signal import welch
from src.calculators import calc_back, radon_process, find_main_directions, calc_autocorr, calc_dispersion


class Back:
    """
    класс
    """

    def __init__(self, data, radon_time: int, fourier_time: int, start_index: int,
                 max_index: int, circle_radon: np.ndarray, threshold=0.5, resolution=4096, size_square=720,
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
        self.ang_std = 0.
        self.speed = float(np.median(data.variables["sog_radar"][start_index: start_index + fourier_time]))
        self.start_index = start_index
        self.threshold = threshold
        self.radon_array = np.zeros(shape=(radon_time, size_square_px, 180), dtype=np.double)

        # full recorded number of turns
        self.max_index = max_index

    def calc_radon(self, back_flag: int):
        """

        """
        # try cut data from azimuth with maximum dispersion
        back_cart_3d_rad, self.ang_std = calc_back(self.data, self.radon_time,
                                                   self.start_index, self.max_index, self.resolution, self.size_square,
                                                   self.size_square_px, back_flag, 0)
        print("back for radon done")
        print("ang_std >> ", self.ang_std)

        self.radon_array = radon_process(back_cart_3d_rad, self.threshold, self.circle_radon)

        return self.radon_array

    def directions_search(self, peak_numbers: int, peak_window: int) -> list:

        angles, direct = find_main_directions(self.radon_array, peak_numbers, peak_window)
        print("obtained direction >> ", angles)
        print("obtained directs >> ", direct)

        if np.abs(direct[0]) < 0.1:  # if data so noisy that we can't determine directions
            # try to cut data from 270 azimuth
            self.calc_radon(1)
            angles2, direct2 = find_main_directions(self.radon_array, peak_numbers, peak_window)
            print("bad angles, new directions >> ", angles2, direct2)
            if np.abs(direct2[0]) > np.abs(direct[0]):
                angles = angles2
                direct = direct2

        direct_std = calc_autocorr(self.radon_array[:, :, self.ang_std % 180])
        print("direct std", direct_std)
        if direct_std < 0:
            if self.ang_std > 180:
                self.ang_std -= 180
            else:
                self.ang_std += 180

        # if the direction obtained by Radon is very different from the largest standard deviation,
        # we give priority to the standard deviation
        if np.abs(direct_std) > np.abs(direct[0]) and np.abs(self.ang_std - angles[0]) > 60:
            angles[0] = self.ang_std

        print("final", angles[0])
        return angles

    def calc_fourier(self, angles: list, cut_ind: int, disp_width: int, turn_period: float, name):

        for an in angles:  # loop in every obtained direction (so we can separate different wave systems)

            # but now we compare only main parameters (because buoy)

            if an == angles[0]:
                back_cartesian_four, ang_std = calc_back(self.data, self.fourier_time, self.start_index, self.max_index,
                                                         self.resolution, self.size_square, cut_ind, 2, an)
                print("back for fourier done")
                f, res_s = welch(back_cartesian_four, detrend='linear', axis=0, return_onesided=False)
                print("welch done")

                k_max = 2 * np.pi / self.size_square * cut_ind

                return calc_dispersion(name, res_s, self.speed, disp_width, turn_period, k_max)
