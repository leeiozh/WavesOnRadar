import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def func(x, A, B) \
        :
    return A + B * np.sqrt(x)


def rmse(array1, array2):
    print(77, np.sqrt(array1.shape[0]))
    print(88, np.linalg.norm(array1 - array2))
    print(array1 - array2)
    return np.linalg.norm(array1 - array2) / np.sqrt(array1.shape[0])


number = '21'

PATH = '/storage/kubrick/ezhova/WavesOnRadar/'

df = pd.read_csv(PATH + "sheets/stations_data13.csv", delimiter=",")
df2 = pd.read_csv(PATH + "sheets/stations_data12.csv", delimiter=",")

IND_AI58 = df["radar_m0"].shape[0]
msk1 = ~np.ma.masked_invalid(df['nobrak']).mask
msk2 = (~np.ma.masked_invalid(df2['sat_swh']).mask & ~np.ma.masked_invalid(df2['nobrak']).mask)

con_m0 = np.append(df["radar_m0"][msk1].to_numpy(), df2["radar_m0"][msk2].to_numpy())
con_swh = np.append(df["buoy_swh"][msk1].to_numpy(), df2["sat_swh"][msk2].to_numpy())

popt1, pcov1 = curve_fit(func, df["radar_m0"][msk1], df["buoy_swh"][msk1])
popt2, pcov2 = curve_fit(func, df2["radar_m0"][msk2], df2["sat_swh"][msk2])
popt, pcov = curve_fit(func, con_m0, con_swh)
print(popt1, popt2)

fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.set_xlim(0., 4.5)
axs.set_ylim(0., 4.5)
axs.grid(linestyle=':')

axs.scatter(con_swh, func(con_m0, *popt), edgecolor='black', s=70, label='satellite')
axs.text(3, 1, 'corr =' + str(round(np.corrcoef(con_swh, func(con_m0, *popt))[0, 1], 2)))
axs.text(3, 1.2, 'rmse =' + str(round(rmse(con_swh, func(con_m0, *popt)), 2)))

axs.scatter(df["buoy_swh"][:IND_AI58], func(df["radar_m0"][:IND_AI58], *popt), edgecolor='black', s=70, label='buoy')
axs.legend()
axs.plot([0., 5], [0., 5], color='black')
axs.set_xlabel("SWH by BUOY/SATELLITE, m")
axs.set_ylabel("SWH by RADAR, m")
plt.savefig(PATH + 'plots/swh_all_' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.set_xlim(0., 4.5)
axs.set_ylim(0., 4.5)
axs.grid(linestyle=':')

axs.scatter(df["buoy_swh"][msk1], func(df["radar_m0"][msk1], *popt1), edgecolor='black', s=70)
axs.plot([0., 5], [0., 5], color='black')
axs.text(1, 3,
         'corr =' + str(
             round(np.corrcoef(df["buoy_swh"][msk1], func(df["radar_m0"][msk1], *popt1))[0, 1], 2)))
axs.text(1, 3.2, 'rmse =' + str(round(rmse(df["buoy_swh"][msk1], func(df["radar_m0"][msk1], *popt1)), 2)))
axs.set_xlabel("SWH by BUOY, m")
axs.set_ylabel("SWH by RADAR, m")
plt.savefig(PATH + 'plots/swh_stat_' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.set_xlim(0., 4.5)
axs.set_ylim(0., 4.5)
axs.grid(linestyle=':')

axs.scatter(df2["sat_swh"][msk2], func(df2["radar_m0"][msk2], *popt), edgecolor='black', s=70)
axs.plot([0., 5], [0., 5], color='black')
axs.text(1, 3,
         'corr =' + str(
             round(np.corrcoef(df2["sat_swh"][msk2], func(df2["radar_m0"][msk2], *popt))[0, 1], 2)))
axs.text(1, 3.2, 'rmse =' + str(round(rmse(df2["sat_swh"][msk2], func(df2["radar_m0"][msk2], *popt)), 2)))
axs.set_xlabel("SWH by SATELLITE, m")
axs.set_ylabel("SWH by RADAR, m")
plt.savefig(PATH + 'plots/swh_notstat_' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.grid(linestyle=':')

axs.scatter(df["buoy_per"][msk1], df["radar_per"][msk1], edgecolor='black', s=70)
axs.plot([0, 20], [0, 20], color='black')
axs.set_xlabel("Period by BUOY, s")
axs.set_ylabel("Period by RADAR, s")
axs.set_xlim(0, 20)
axs.set_ylim(0, 20)
axs.text(12, 3,
         'corr =' + str(round(np.corrcoef(df["buoy_per"].to_numpy()[msk1], df["radar_per"].to_numpy()[msk1])[0, 1], 2)))
axs.text(12, 4, 'rmse =' + str(round(rmse(df["buoy_per"].to_numpy()[msk1], df["radar_per"].to_numpy()[msk1]), 2)))

plt.savefig(PATH + 'plots/per' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(4, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.grid(linestyle=':')

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

print(np.abs(tmp_an - tmp_an_buoy))

axs.scatter(tmp_an_buoy, tmp_an, edgecolor='black', s=70, label='clear data')
axs.plot([0, 360], [0, 360], color='black')

axs.set_xlabel("Direction by BUOY, deg")
axs.set_ylabel("Direction by RADAR, deg")
axs.set_xlim(-20, 380)
axs.set_ylim(-20, 380)
axs.text(20, 270, 'corr =' + str(round(np.corrcoef(tmp_an_buoy, tmp_an)[0, 1], 2)))
axs.text(20, 250, 'rmse =' + str(round(rmse(tmp_an_buoy, tmp_an), 2)))

axs.grid(linestyle=':')
plt.savefig(PATH + 'plots/dir' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.grid(linestyle=':')
axs.scatter(df["speed"][msk1], np.abs(df["buoy_swh"][msk1] - func(df["radar_m0"][msk1], *popt1)) / df["buoy_swh"][msk1],
            label='buoy', edgecolor='black', s=40)
axs.scatter(df2["speed"][msk2],
            np.abs(df2["sat_swh"][msk2] - func(df2["radar_m0"][msk2], *popt)) / df2["sat_swh"][msk2], label='satellite',
            edgecolor='black', s=40)
plt.legend()
axs.set_ylim(-0.25, 1.5)
axs.grid(linestyle=':')
axs.set_xlabel("Speed over ground, m/s")
axs.set_ylabel(r"$\frac{|SWH by SAT/BUOY - SWH by RADAR|}{SWH by SAT/BUOY}$")
plt.savefig(PATH + 'plots/speed2' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.3, wspace=.3)
axs.get_xaxis().set_tick_params(which='both', direction='in')
axs.get_yaxis().set_tick_params(which='both', direction='in')
plt.rc('axes', axisbelow=True)
axs.grid(linestyle=':')
axs.scatter(df["wind_speed"][msk1], np.abs(df["buoy_swh"][msk1] - func(df["radar_m0"][msk1], *popt1)) / df["buoy_swh"][msk1],
            label='buoy', edgecolor='black', s=40)
axs.scatter(df2["wind_speed"][msk2],
            np.abs(df2["sat_swh"][msk2] - func(df2["radar_m0"][msk2], *popt)) / df2["sat_swh"][msk2], label='satellite',
            edgecolor='black', s=40)
plt.legend()
#axs.set_ylim(-0.25, 1.5)
axs.grid(linestyle=':')
axs.set_xlabel("Wind speed, m/s")
axs.set_ylabel(r"$\frac{|SWH by SAT/BUOY - SWH by RADAR|}{SWH by SAT/BUOY}$")
plt.savefig(PATH + 'plots/wind' + number + '.png', bbox_inches='tight', dpi=1000)
plt.show()
