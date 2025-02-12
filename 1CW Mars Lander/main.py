import sys
import genfigs
from genfigs import Integrator
import numpy as np

def main():

    """genfigs.EffectOfTime(show=False)
    eod = genfigs.EffectOfTime(show=False)
    eod.start()"""

    generate_3d_plot(x_init=np.array([1, 1, 1]),
                     v_init = np.array([10, 0, 0]),
                     centre = np.array([0, 0, 0]),
                     t_max = 5,
                     dt = 0.01,
                     i_name = 'verlet',
                     f_name = 'grav',
                     file_name= 'First')

    generate_3d_plot(x_init=np.array([1, 1, 1]),
                     v_init=np.array([0, 0, 0]),
                     centre=np.array([0, 0, 0]),
                     t_max=5,
                     dt=0.01,
                     i_name='verlet',
                     f_name='grav',
                     file_name='Second')

    return


def generate_3d_plot(x_init, v_init, dt, t_max, centre, i_name, f_name, file_name):

    td = genfigs.ThreeDimensionalPlot(x_init, v_init, dt, t_max,  centre, show=True)

    if i_name == 'euler' and f_name == 'grav':
        td.i.euler(f_func=Integrator.grav)

    elif i_name == 'euler' and f_name == 'spring':
        td.i.euler(f_func=Integrator.spring)

    elif i_name == 'verlet' and f_name == 'grav':
        td.i.verlet(f_func=Integrator.grav)

    elif i_name == 'verlet' and f_name == 'spring':
        td.i.verlet(f_func=Integrator.spring)

    else:
        print('i_name and/or f_name error')
        sys.exit()

    td.start()
    td.generate_file(file_name=file_name)

if __name__ == "__main__":
    main()
