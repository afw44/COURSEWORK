import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import square
import sys


df1 = pd.read_csv('data/5.csv', delimiter=',')
df2 = pd.read_csv('data/6.csv', delimiter=',')
df3 = pd.read_csv('data/7.csv', delimiter=',')



def plot(ax,df,c,l):

    x = df['Frequency'].tolist()
    y = df['Amplitude'].tolist()

    for i in range(len(x)):

        if i==0:
            ax.plot([x[i],x[i]],[0, y[i]], color=c, label=l)
        else:
            ax.plot([x[i],x[i]],[0, y[i]], color=c)

        ax.scatter([x[i]],[y[i]], marker='x', color=c)

    print(x)
    print(y)
    return



def makeplot(y):
    fig, ax = plt.subplots()
    ax.set_xlim(0,20)
    ax.set_ylim(0, 1.5)

    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('AM Spectra Peaks')

    plot(ax,df1,'red', 'm=1.00')
    plot(ax,df2,'blue', 'm=0.61')
    if y:
        plot(ax,df3,'green', 'suppressed carrier')

    ax.legend()

    if not y:
        fig.savefig('figs/AM spectra 1')
    else:
        fig.savefig('figs/AM spectra 2')

makeplot(False)
makeplot(True)


plt.show()