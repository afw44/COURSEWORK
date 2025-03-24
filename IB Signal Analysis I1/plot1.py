import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import square
import sys

def theoretical_fourier_coefs(df1, func, N):

    df1['a'] = df1.apply(lambda row: (row['Amplitude'] /  df1['Amplitude'][0]), axis=1)
    df1['exp'] = df1.apply(func, axis=1)
    df1['exp'] = df1.apply(lambda row: (row['exp'] / df1['exp'][0]), axis=1)


    df1['e'] = df1.apply(lambda row: 100*((row['a'] / row['exp']) - 1), axis=1)

    x1 = np.linspace(1, max(df1['Frequency']), 100)
    xlims =[min(df1['Frequency']),max(df1['Frequency'])]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    ax1.scatter(df1['Frequency'], df1['a'], label='Expected from series')
    ax1.scatter(df1['Frequency'], df1['exp'],c='r',zorder=-1, label='Actual from experiment')
    ax2.scatter(df1['Frequency'], df1['e'], label='Error', marker='x',c='black')
    ax2.plot(xlims,[0,0], linestyle='--', color='black',zorder=-2)

    ax1.plot(x1, 1 / (x1**(N+1)), linestyle='--', color='black', zorder=-2)

    ax1.set_ylim(0,1.5)
    ax1.set_xlim(xlims)
    ax2.set_xlim(xlims)
    ax2.set_ylabel('Percentage Error')
    ax1.set_ylabel('Relative Amplitude of Peak')
    ax1.set_xlabel('Frequency (kHz)')
    ax1.grid('on')
    ax2.set_ylim([-3*max(abs(df1['e'])),1.5*max(abs(df1['e']))])

    ax1.legend()
    ax2.legend(loc='upper left')
    fig.tight_layout()

    fig.savefig(f'figs/fourier_coefs_{N}.png')

    print(df1)

    return

f1 = lambda row : 1/row['Frequency']
theoretical_fourier_coefs(pd.read_csv('data/q3.csv', delimiter=','), f1, 0)

f2 = lambda row : np.abs((8/np.pi**2) *((-1)**(0.5*(int(row['Frequency']))-1))/((row['Frequency'])**2))
theoretical_fourier_coefs(pd.read_csv('data/q4.csv', delimiter=','),f2, 1)

plt.show()
