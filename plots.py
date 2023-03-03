import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def func(x, A, B):
    return A + B * np.sqrt(x)


PATH = '/storage/kubrick/ezhova/WavesOnRadar/'

fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)

axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)

df = pd.read_csv("/storage/kubrick/ezhova/WavesOnRadar/sheets/stations_data13.csv", delimiter=",")

IND_AI58 = 30

popt, pcov = curve_fit(func, df["radar_m0"][:IND_AI58], df["buoy_swh"][:IND_AI58])

axs.scatter(df["buoy_swh"][:IND_AI58], func(df["radar_m0"][:IND_AI58], *popt), edgecolor='black', s=70)
# axs.scatter(df["buoy_swh"] * (1 - mask), func(df["radar_m0"] * (1 - mask), *popt), edgecolor='black', s=70,
##            label='noisy data')
# axs.legend()
axs.plot([0., 5], [0., 5], color='black')
#
axs.text(3, 1,
         'corr =' + str(round(np.corrcoef(df["buoy_swh"][:IND_AI58], func(df["radar_m0"][:IND_AI58], *popt))[0, 1], 2)))
axs.text(3, 1.2, 'rmse =' + str(round(
    np.linalg.norm(df["buoy_swh"][:IND_AI58] - func(df["radar_m0"][:IND_AI58], *popt)) / np.sqrt(
        df[:IND_AI58].shape[0]), 2)))
#
axs.set_xlabel("SWH by BUOY, m")
axs.set_ylabel("SWH by RADAR, m")
axs.set_xlim(0., 4.5)
axs.set_ylim(0., 4.5)
#
tmp_an = df["radar_an"].to_numpy()[:30]
tmp_an_buoy = df["buoy_ang"].to_numpy()[:30]

for i in range(tmp_an.shape[0]):
    if tmp_an[i] - tmp_an_buoy[i] > 180:
        if tmp_an[i] > 180:
            tmp_an[i] -= 180
        else:
            tmp_an[i] += 180
    elif tmp_an_buoy[i] - tmp_an[i] > 180:
        if tmp_an[i] > 180:
            tmp_an[i] -= 180
        else:
            tmp_an[i] += 180

# print(np.abs(tmp_an - tmp_an_buoy))

# axs.scatter(tmp_an_buoy, tmp_an, edgecolor='black', s=70, label='clear data')
# axs.scatter(df["buoy_ang"][-11 + IND_AI58:-6 + IND_AI58], tmp_an[-11 + IND_AI58:-6 + IND_AI58], color='orange',
#             edgecolor='black', s=70, label='noisy data')
# axs.legend()
# axs.plot([0, 360], [0, 360], color='black')

# axs.set_xlabel("Direction by BUOY, deg")
# axs.set_ylabel("Direction by RADAR, deg")
# axs.set_xlim(-20, 380)
# axs.set_ylim(-20, 380)
# axs.text(20, 270, 'corr =' + str(round(np.corrcoef(tmp_an_buoy, tmp_an)[0, 1], 2)))
# axs.text(20, 250, 'rmse =' + str(round(np.linalg.norm(tmp_an_buoy - tmp_an) / np.sqrt(tmp_an_buoy.shape[0]), 2)))

# axs.scatter(df["buoy_per"], df["radar_per"], edgecolor='black', s=70)
# axs.legend()
# axs.plot([0, 20], [0, 20], color='black')
# axs.set_xlabel("Period by BUOY, s")
# axs.set_ylabel("Period by RADAR, s")
# axs.set_xlim(0, 20)
# axs.set_ylim(0, 20)
# axs.text(12, 3, 'corr =' + str(round(np.corrcoef(df["buoy_per"], df["radar_per"])[0, 1], 2)))
# axs.text(12, 4,
#          'rmse =' + str(round(np.linalg.norm(df["buoy_per"] - df["radar_per"]) / np.sqrt(df["radar_per"].shape[0]), 2)))

axs.grid(linestyle=':')
plt.savefig(PATH + 'plots/swh10.png', bbox_inches='tight', dpi=1000)
plt.show()
