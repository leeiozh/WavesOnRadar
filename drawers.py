import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def make_anim_radon(array, frames, name):
    def anim(i):
        # print(i)
        im.set_array(array[i])
        #line.set_data(np.arange(180), 10 * np.std(array[i], axis=0) - 700)

        #line2.set_data(50 + 2 * array[:, np.argmax(np.std(array[i], axis=0))], np.linspace(-360, 360, 384))

        # axs.text(100, 200, str(i), color='black', bbox=dict(boxstyle='round', facecolor='white'))
        return [im]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='w', edgecolor='k')
    im = axs.imshow(array[0], origin='lower', cmap='gnuplot', extent=[0, 180, -360, 360], aspect=0.25,
                    interpolation='None')  # extent=[0, 180, 0, 720], aspect=0.25,
    #line, = axs.plot(np.arange(180), 10 * np.std(array[0], axis=0) - 700, color='black')
    #line2, = axs.plot(50 + 2 * np.std(array[0], axis=1), np.linspace(-360, 360, 384), color='white')
    animation = FuncAnimation(fig, anim, frames=frames, interval=100000, blit=True)
    writer = PillowWriter(fps=1)

    # axs.set_xlabel(r"Угол $\theta$, град")
    # axs.set_ylabel(r"Расстояние $\rho$, м")
    animation.save(str(name) + ".gif", writer=writer)


def make_anim_back(array, frames, name):
    def anim(i):
        # print(i)
        im.set_array(array[i])
        # axs.text(100, 200, str(i), color='black', bbox=dict(boxstyle='round', facecolor='white'))
        return [im]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='w', edgecolor='k')
    im = axs.imshow(array[0], origin='lower', cmap='Greys',
                    interpolation='None')  # extent=[0, 180, 0, 720], aspect=0.25,
    animation = FuncAnimation(fig, anim, frames=frames, interval=100000, blit=True)
    writer = PillowWriter(fps=1)

    # axs.set_xlabel(r"Угол $\theta$, град")
    # axs.set_ylabel(r"Расстояние $\rho$, м")
    animation.save("gifs/" + str(name) + ".gif", writer=writer)


def make_shot(array, name):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='w', edgecolor='k')
    axs.imshow(array, origin='lower', cmap='Greys', interpolation='None')
    plt.savefig("pics/" + str(name) + ".png", dpi=300)


def make_anim_plot(array, name, frames):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='w', edgecolor='k')

    x = np.arange(0, 180)
    line, = axs.plot(x, array[0])
    axs.set_xlabel(r"Угол $\theta$, град")
    axs.set_ylabel(r"Расстояние $\rho$, м")
    axs.grid()

    def anim(i):
        # print(i)
        axs.plot(x, array[i])
        # axs.set_xlabel(r'Угол \theta, град')
        # axs.set_ylabel(r'\int R(\theta, s)ds / \int ds')
        # axs.text(100, 200, str(i), color='black', bbox=dict(boxstyle='round', facecolor='white'))
        return line,

    animation = FuncAnimation(fig, anim, frames=frames, interval=10000, blit=True)
    writer = PillowWriter(fps=2.5)

    animation.save(str(name) + ".gif", writer=writer)