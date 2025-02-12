import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class Series:

    def __init__(self, data):

        self.data = data
        self.lindata = None
        self.label = None
        self.lowlim = None
        self.highlim = None
        self.P = None
        self.grad = None
        self.loadlim = None

    def populate_lindata(self):

        self.lindata = self.data[np.argwhere(
            np.logical_and(
                (self.lowlim < self.data[:,0]),
                (self.highlim > self.data[:,0]))).flatten()]

    def fit_lindata(self):

        self.P = np.poly1d(np.polyfit(self.lindata[:,0], self.lindata[:,1], deg=1))
        print(self.P)

    def scatter_all(self, ax, sparsity=1):

        d = self.data[0:-1:int(1/sparsity)]

        ax.scatter(d[:, 0], d[:, 1],
                   marker='x', color='black', zorder=0)

    def add_lims(self, ax):

        ax.plot([self.lowlim, self.lowlim],
                [np.min(self.data[:, 1]), self.P(self.lowlim)],
                color='red',
                linestyle='--')

        ax.plot([self.highlim, self.highlim],
                [np.min(self.data[:, 1]), self.P(self.highlim)],
                color='red',
                linestyle='--')

    def add_fitline(self, ax):

        X = np.linspace(np.min(self.data[:, 0]), np.max(self.data[:, 0]), 100)

        ax.plot(X, self.P(X), linewidth=3, zorder=1, color='green')

        self.loadlim = self.P(self.highlim)
        ax.scatter([self.lowlim, self.highlim],
                   [self.P(self.lowlim), self.P(self.highlim)],
                   color='red', s=50)

        self.grad = self.P[1]

    def add_text(self, ax):

        t = f"For {self.label}, \ndF/dx was {round(self.P[1],2)} N/mm"

        ax.text(np.max(self.data[:, 0])-5.3, np.min(self.data[:, 1])+ 12, t,
                fontsize=14, color='black', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))


class Helper:

    def __init__(self):
        self.all_dfs = None
        self.all_linregs = None

    def reader(self):

        df1 = pd.read_csv('data/mild_steel_1.csv')[1:].to_numpy(dtype=float)[:,1:]
        df2 = pd.read_csv('data/aluminium_1.csv')[1:].to_numpy(dtype=float)[:,1:]
        df3 = pd.read_csv('data/PMMA_1.csv')[1:].to_numpy(dtype=float)[:,1:]

        self.all_dfs = [Series(df1),
                        Series(df2),
                        Series(df3)]

        lmat = np.array([[1.0, 6.4],
                         [1.0, 4.5],
                         [1.0, 6.7]])

        labels = ['Mild Steel', 'Aluminium', 'PMMA']
        densities = [7860, 2710, 1170]

        for i,df in enumerate(self.all_dfs):

            df.label = labels[i]
            df.lowlim = lmat[i][0]
            df.highlim = lmat[i][1]
            df.rho = densities[i]

            df.populate_lindata()
            df.fit_lindata()

    def plotter(self, show_lims=False):

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7,14))
        fig.suptitle('Force by Displacement for each sample\n', fontsize=16)
        for i,ax in enumerate(axs.flatten()):

            df = self.all_dfs[i]

            ax.set_xlim(np.min(df.data[:, 0]), np.max(df.data[:, 0]))
            ax.set_ylim(np.min(df.data[:, 1]), np.max(df.data[:, 1]))

            ax.set_xlabel('Displacement(mm)', fontsize=10)
            ax.set_ylabel('Force (F)', fontsize=10)
            ax.set_title(df.label, fontsize=16)

            df.scatter_all(ax, sparsity=0.03)
            df.add_fitline(ax)
            df.add_lims(ax)
            df.add_text(ax)

        fig.tight_layout()
        fig.savefig('Output Figure')

    def tables(self):

        df1 = pd.DataFrame([['Mild Steel',  25,1.8,0, 1, 0],
                           ['Aluminium',    25,2.85,0, 1, 0],
                           ['PMMA',         25,5.18,0, 1, 0]])

        df1.columns = ['Material', 'b (mm)','h (mm)','I / 10^-12 m^4',  'F-x Gradient (N/mm)', "Young's Modulus (GPa)"]

        for i,s in enumerate(self.all_dfs):

            df1.iloc[i,3] =  (df1.iloc[i,1] * df1.iloc[i,2]**3 /(12*10**12))
            df1.iloc[i,4] = round(s.grad,3)
            df1.iloc[i,5] =  round(1000 * ((0.1**2) / 48) * (df1.iloc[i,4] / df1.iloc[i,3]) /10**9,3)

        df2 = pd.DataFrame([['Mild Steel',  25,1.8,0, 1],
                           ['Aluminium',    25,2.85,0, 1],
                           ['PMMA',         25,5.18,0, 1]])

        df2.columns = ['Material', 'y-max (mm)', 'I / 10^-12 m^4','Limiting Force (N)', 'Limiting Stress (MPa)']

        for i,s in enumerate(self.all_dfs):

            df2.iloc[i,1] = df1.iloc[i,2]/2
            df2.iloc[i,2] =  round(df1.iloc[i,3] * 10**12,2)
            df2.iloc[i,3] = round(s.loadlim,3)
            df2.iloc[i,4] = (0.1/4) * (df2.iloc[i,3] * df2.iloc[i,1]) / (df2.iloc[i,2]) * 10**3

        df3 = pd.DataFrame([['Mild Steel', 0, 0, 0],
                            ['Aluminium', 0, 0, 0],
                            ['PMMA', 0, 0, 0]])

        df3.columns = ['Material', 'E (GPa)', 'Yield Stress (MPa)', 'Metric']

        for i,s in enumerate(self.all_dfs):

            df3.iloc[i, 1] = df1.iloc[i, -1]
            df3.iloc[i,2] = df2.iloc[i,-1]

            df3.iloc[i,3] = df3.iloc[i,2]**2 / df3.iloc[i,1]

        print(df1)
        print()
        print(df2)
        print()
        print(df3)

        df1.to_csv('output_tables/table1.csv', index=False)
        df2.to_csv('output_tables/table2.csv', index=False)
        df3.to_csv('output_tables/table3.csv', index=False)

def main():

    H = Helper()
    H.reader()
    H.plotter(show_lims=True)
    H.tables()

if __name__ == '__main__':
    main()
