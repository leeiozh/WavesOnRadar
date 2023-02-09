import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def func(x, A, B):
    return A + B * np.sqrt(x)


fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)

axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)

df = pd.read_csv("/storage/kubrick/ezhova/WavesOnRadar/sheets/stations_data11.csv", delimiter=",")

popt, pcov = curve_fit(func, df["radar_m0"][:-12], df["buoy_swh"][:-12])

# popt = [1.02998521, 12.59410359]

# axs.scatter(df["radar_m0"], df["buoy_swh"])
# axs.plot(np.linspace(0, 0.3, 10), func(np.linspace(0, 0.3, 10), *popt))
mask = np.ones(df.shape[0])
mask[-12:] = np.zeros(12)
# mask[-9:-7] = np.array([1, 1])
# mask[-5] = 1

axs.scatter(df["buoy_swh"], func(df["radar_m0"], *popt), edgecolor='black', s=70, label='clear data')
axs.scatter(df["buoy_swh"] * (1 - mask), func(df["radar_m0"] * (1 - mask), *popt), edgecolor='black', s=70,  label='noisy data')
axs.legend()
axs.plot([0.5, 5], [0.5, 5], color='black')

axs.text(3, 1, 'corr =' + str(
   round(np.corrcoef(df["buoy_swh"][:-12], func(df["radar_m0"], *popt)[:-12])[0, 1], 2)))
axs.text(3, 1.2, 'rmse =' + str(
   round(np.linalg.norm(df["buoy_swh"][:-12] - func(df["radar_m0"], *popt)[:-12]) / np.sqrt(df[:-12].shape[0]), 2)))

axs.set_xlabel("SWH by BUOY, s")
axs.set_ylabel("SWH by RADAR, s")
axs.set_xlim(0.5, 4.5)
axs.set_ylim(0.5, 4.5)

# axs.scatter(df["buoy_ang"], df["radar_an"], edgecolor='black', s=70, label='clear data')
# axs.scatter(df["buoy_ang"][-11:-1], df["radar_an"][-11:-1], color='orange', edgecolor='black', s=70, label='noisy data')
# axs.legend()
# axs.plot([0, 360], [0, 360], color='black')
#
# axs.set_xlabel("Direction by BUOY, deg")
# axs.set_ylabel("Direction by RADAR, deg")
# axs.set_xlim(-20, 380)
# axs.set_ylim(-20, 380)
# axs.text(20, 270, 'corr =' + str(
#     round(np.corrcoef(df["buoy_ang"][:-12], df["radar_an"][:-12])[0, 1], 2)))
# axs.text(20, 250, 'rmse =' + str(
#     round(np.linalg.norm(df["buoy_ang"][:-12] - df["radar_an"][:-12]) / np.sqrt(df[:-12].shape[0]), 2)))

# axs.scatter(df["buoy_per"], df["radar_per"], edgecolor='black', s=70, label='clear data')
# axs.scatter(df["buoy_per"] * (1 - mask), df["radar_per"] * (1 - mask), edgecolor='black', s=70, label='noisy data')
# axs.legend()
# axs.plot([5, 16], [5, 16], color='black')
#
# axs.set_xlabel("Period by BUOY, s")
# axs.set_ylabel("Period by RADAR, s")
# axs.set_xlim(5, 16)
# axs.set_ylim(5, 16)
# axs.text(12, 6, 'corr =' + str(round(np.corrcoef(df["buoy_per"][:-12], df["radar_per"][:-12])[0, 1], 2)))
# axs.text(12, 7, 'rmse =' + str(round(
#     np.linalg.norm(df["buoy_per"][:-12] - df["radar_per"][:-12]) / np.sqrt(df["radar_per"][:-12].shape[0]), 2)))

axs.grid(linestyle=':')
plt.savefig('/storage/kubrick/ezhova/WavesOnRadar/plots/swh.png', bbox_inches='tight', dpi=1000)
plt.show()
