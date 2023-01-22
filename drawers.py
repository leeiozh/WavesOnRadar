from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def make_anim(array, frames, name):
    def anim(i):
        # print(i)
        im.set_array(array[i])
        # axs.text(100, 200, str(i), color='black', bbox=dict(boxstyle='round', facecolor='white'))
        return [im]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='w', edgecolor='k')
    im = axs.imshow(array[0], origin='lower', cmap='gnuplot',
                    interpolation='None')  # extent=[0, 180, 0, 720], aspect=0.25,
    animation = FuncAnimation(fig, anim, frames=frames, interval=100000, blit=True)
    writer = PillowWriter(fps=1)

    # axs.set_xlabel(r"Угол $\theta$, град")
    # axs.set_ylabel(r"Расстояние $\rho$, м")
    animation.save(str(name) + ".gif", writer=writer)


def make_shot(array, name):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7), facecolor='w', edgecolor='k')
    axs.imshow(array, origin='lower', cmap='gnuplot', interpolation='None')
    plt.savefig(str(name) + ".png", dpi=300)
