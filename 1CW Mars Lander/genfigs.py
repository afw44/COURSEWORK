import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
from datetime import datetime
import os
from os import listdir


class EffectOfTime:

    def __init__(self, show=False):

        self.fig, self.axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5),
                                layout="constrained")

        self.dt_array = np.array([[0.05, 0.01],
                            [0.001, 0.0001]])

        self.x_init = np.array([10])
        self.v_init = np.array([0])
        self.centre = np.array([0])

        self.t_max = 5
        self.dt = None

        self.x_array = None
        self.v_array = None
        self.t_array = None

        self.i = None

    def start(self):

        for row in range(2):
            for col in range(2):

                self.dt = self.dt_array[row][col]

                self.i = Integrator(self.x_init, self.v_init,
                            self.dt, self.t_max, self.centre).euler(f_func=Integrator.spring)

                self.x_array = self.i.x_array
                self.v_array = self.i.v_array
                self.t_array = self.i.t_array

                self.axs[row, col].plot(self.t_array, self.x_array)
                self.axs[row, col].set_title(f"dt = {self.dt}")

                self.axs[row, col].plot([self.t_array[0], self.t_array[-1]], [10, 10], c='r')
                self.axs[row, col].plot([self.t_array[0], self.t_array[-1]], [-10, -10], c='r')

            self.fig.suptitle(f'Effect of dt on simulations')

        return

class ThreeDimensionalPlot:

    def __init__(self, x_init, v_init, dt, t_max, centre, show=True):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line, = self.ax.plot([], [], [], lw=4, c='black')

        self.dt = dt
        self.t_max = t_max
        self.centre = centre

        self.x_array = None
        self.v_array = None
        self.t_array = None

        self.i = None
        self.anim = None

        self.i = Integrator(x_init, v_init, dt, t_max, centre).initialise()

        self.x_array = self.i.x_array
        self.v_array = self.i.v_array
        self.t_array = self.i.t_array

    def start(self):

        sparse_indices = np.arange(0, len(self.t_array), step=int(len(self.t_array) / 100))

        spatial_x = self.x_array[:, 0][sparse_indices]
        spatial_y = self.x_array[:, 1][sparse_indices]
        spatial_z = self.x_array[:, 2][sparse_indices]

        self.ax.scatter([self.centre[0]],[self.centre[1]],[self.centre[2]], s=5)

        self.ax.plot(spatial_x, spatial_y, spatial_z, color='red', linestyle='dotted', alpha=0.5)

        self.anim = animation.FuncAnimation(self.fig,
                                            self.animate,
                                            init_func=self.init,
                                            fargs=(spatial_x, spatial_y, spatial_z),
                                            frames=len(spatial_x),
                                            interval=200,
                                            repeat_delay=5,
                                            blit=True)

        return

    def init(self):
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        return self.line,

    def animate(self, i, X, Y, Z):
        self.line.set_data(X[:i], Y[:i])
        self.line.set_3d_properties(Z[:i])
        return self.line,

    def generate_file(self, file_name='No Filename Declared'):
        header = file_name

        for file in listdir('AnimatedPlots'):
            if file.startswith(header):
                try:
                    os.remove(f'AnimatedPlots/{file}')
                    print(f"File '{file}' deleted successfully.")
                except FileNotFoundError:
                    print(f"File '{file}' not found.")

        fstring = '%d_%m %H:%M:%S'
        path = f'AnimatedPlots/{header} {datetime.now().strftime(fstring)}.mp4'
        self.anim.save(path, fps=30)

        return

class Integrator:

    def __init__(self, x_init, v_init, dt, t_max, centre):

        self.x_init = x_init
        self.v_init = v_init
        self.dt = dt
        self.t_max = t_max
        self.centre = centre

        self.x_array = None
        self.v_array = None
        self.t_array = None

        self.current_x = x_init
        self.current_v = v_init

    def initialise(self):

        assert np.shape(self.x_init)[0] == np.shape(self.v_init)[0]

        dim = np.shape(self.x_init)[0]

        t_array = np.arange(0, self.t_max, self.dt)

        x_array = np.zeros((len(t_array), dim))
        v_array = np.zeros((len(t_array), dim))

        x_array[0] = self.x_init
        v_array[0] = self.v_init

        self.x_array, self.v_array, self.t_array \
            = x_array, v_array, t_array

        return self

    def verlet(self, f_func):

        self.initialise()

        for i in range(1, len(self.t_array)):

            if i == 1:

                self.current_x = (self.x_array[i - 1]
                                  + 0.5 * f_func(self, self.x_array[i]) * self.dt ** 2)
                self.x_array[i] = self.current_x

                self.current_v = (self.v_array[i - 1]
                                  - self.dt * f_func(self, self.x_array[i]))
                self.v_array[i] = self.current_v


            else:

                self.current_x = (2 * self.x_array[i - 1]
                                  - self.x_array[i - 2]
                                  + (self.dt ** 2) * f_func(self, self.x_array[i - 1]))
                self.x_array[i] = self.current_x

                self.current_v = ((self.x_array[i] -
                                  self.x_array[i - 2])
                                  / (2 * self.dt))
                self.v_array[i] = self.current_v

        return self

    def euler(self, f_func):

        self.initialise()

        for i in range(1, len(self.t_array)):
            self.v_array[i] = self.v_array[i - 1] + self.dt * f_func(self, self.x_array[i - 1])
            self.x_array[i] = self.x_array[i - 1] + self.dt * self.v_array[i]

        return self

    def spring(self, _k=None, _m=None):

        _k = 1
        m = 1

        return - _k * (self.current_x - self.centre) / m

    def grav(self, _m=None, _g=None):

        _m = 6.43 * 10 ** 6
        _g = 6.67 * 10 * -18

        return - _g * _m / ((self.current_x - self.centre) ** 2)