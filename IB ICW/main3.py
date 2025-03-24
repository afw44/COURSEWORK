import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def solve_eigen(K, M):

    eigvals, eigvecs = la.eigh(K, M)
    idx = np.argsort(eigvals)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    freqs = np.sqrt(np.maximum(eigvals, 0))

    pred_modes_norm = np.zeros_like(eigvecs)
    for j in range(eigvecs.shape[1]):
        pred_modes_norm[:, j] = eigvecs[:, j] / eigvecs[:, j][0]

    return freqs, pred_modes_norm

def makemats(ms, ks):
    [m1, m2, m3] = ms
    [k1, k2, k3] = ks

    M0 = np.diag([m1, m2, m3])
    K = np.array([[k1 + k2, -k2, 0],
                  [-k2, k2 + k3, -k3],
                  [0, -k3, k3]])

    return M0,K

def plotter(ax1,ax2,ms,ks,exp):

    M0,K = makemats(ms,ks)
    M = M0.copy()
    M += np.diag(exp.dm)

    pred_freqs, pred_modes = solve_eigen(K,M)

    print(pred_freqs)
    print(pred_modes)

    ax1.set_title(exp.dm)
    ax1.set_xlim(0, max(pred_freqs) * 1.1)
    ax1.set_yticks([])
    ax1.set_ylim(-.5, .5)
    ax1.plot([0,max(pred_freqs) * 1.1], [0,0], color='black', linestyle='--')

    for i,q in enumerate(pred_freqs):
        ax1.plot([pred_freqs[i], pred_freqs[i]], [-.5, 0], c='r', linestyle='dashed')
        ax1.plot([exp.freqs[i],exp.freqs[i]],[0,0.5],c='b',linestyle='dashed')

        ax1.plot([pred_freqs[i],pred_freqs[i]],[0,.35],c='r')
        ax1.plot([exp.freqs[i],exp.freqs[i]],[-.35,0],c='b')

        ax1.scatter(pred_freqs[i],0.35, marker='x', c='r')
        ax1.scatter(exp.freqs[i], -0.35, marker='x', c='b')

    ax2.set_title(exp.dm)
    ax2.set_ylim(-3.2,3.2)
    ax2.set_xlim(0.5,4.5)
    ax2.set_xticks([])

    ax2.plot([0.5,4.5],[0,0], linestyle='dashed',c='black')
    cs=['red','blue','green']

    for i,q in enumerate(pred_modes):
        p = np.zeros((1, 4))
        q = np.zeros((1, 4))
        p[0][1:4] = pred_modes.transpose()[i]
        q[0][1:4] = exp.modes.transpose()[i]

        ax2.plot([1,2,3,4], p[0], linestyle='dotted',c=cs[i])
        ax2.scatter([1,2,3,4], p[0], marker='x', c=cs[i])

        ax2.plot([1, 2, 3, 4], q[0], linestyle='solid', c=cs[i])
        ax2.scatter([1, 2, 3, 4], q[0], marker='o', c=cs[i])

    return

def objective_function(p, experiments):

    m1, m2, m3, k1, k2, k3 = p
    M0,K = makemats([m1,m2,m3],[k1,k2,k3])
    errors = []

    for exp in experiments:

        # Make perturbed mass matrix M = M_0 + E(dm)
        M = M0.copy()
        M += np.diag(exp.dm)

        # Predicted freqs and modeshapes for the current guesses,
        # by solving the eigen-equation
        pred_freqs, pred_modes = solve_eigen(K, M)

        # Eigenvalue error
        freq_err = pred_freqs - exp.freqs
        errors.extend(freq_err)

        # Eigenvector error
        for j in range(pred_modes.shape[1]):
            mode_err = pred_modes[:, j] - exp.modes[:, j]
            errors.extend(mode_err)

    return np.array(errors)

class Exp:

    def __init__(self, floor, dm, freqs, modes):

        self.floor = floor
        self.dm = dm
        self.freqs = freqs
        self.modes = modes

def main():
    ms = pd.read_csv('data/mode_shapes.csv', delimiter=',')
    eig = pd.read_csv('data/eigen.csv', delimiter=',')

    eigs = [eig.iloc[3 * i:3 * (i + 1), :].to_numpy().transpose()[0] * 2 * np.pi for i in range(5)]
    mss = [ms.iloc[3 * i:3 * (i + 1), :].to_numpy().transpose() for i in range(5)]

    # Initial guess for parameters: [m1, m2, m3, k1, k2, k3]

    p0 = np.array([1.4, 1.4, 1.4, 3000, 3000, 3000], dtype=float)

    experiments = list()

    experiments.append(Exp(floor=0, dm=[0,      0,      0    ],   freqs=eigs[0], modes=mss[0]))
    experiments.append(Exp(floor=1, dm=[0.406,  0,      0    ],   freqs=eigs[1], modes=mss[1]))
    experiments.append(Exp(floor=2, dm=[0,      0.406,  0    ],   freqs=eigs[2], modes=mss[2]))
    experiments.append(Exp(floor=3, dm=[0,      0,      0.406],   freqs=eigs[3], modes=mss[3]))

    trial_exp = Exp(floor=3, dm=[0.403,      0,      0.406],   freqs=eigs[3], modes=mss[3])

    lower_bounds = np.full(6, 1e-6)  # masses and stiffnesses must be positive
    upper_bounds = np.full(6, np.inf)

    p_opt = least_squares(objective_function, p0, args=(experiments,),
                           bounds=(lower_bounds, upper_bounds),
                           xtol=1e-6, ftol=1e-6, verbose=0).x

    m_est = p_opt[0:3]
    k_est = p_opt[3:6]

    M_x, K_x = makemats(m_est, k_est)

    fig3 = plt.figure(layout="constrained", figsize = (10,10))
    axs3 = list()
    gs = GridSpec(5, 2, figure=fig3)

    axs3.append([fig3.add_subplot(gs[i,1]) for i in range(0,5)])
    axs3.append([fig3.add_subplot(gs[i,0]) for i in range(0,5)])

    print(axs3)


    fig1,axs1 = plt.subplots(2,2, figsize=(10,10))
    fig2,axs2 = plt.subplots(2,2, figsize=(10,10))

    fig1.suptitle('Predicted / Experimental fundamental frequencies')
    fig2.suptitle('Predicted / Experimental modes')

    plotter(axs3[0][0],    axs3[1][0], m_est, k_est, exp=experiments[0])
    plotter(axs3[0][1],    axs3[1][1], m_est, k_est, exp=experiments[1])
    plotter(axs3[0][2],    axs3[1][2], m_est, k_est, exp=experiments[2])
    plotter(axs3[0][3],    axs3[1][3], m_est, k_est, exp=experiments[3])
    plotter(axs3[0][4],    axs3[1][4], m_est, k_est, exp=trial_exp)


    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    np.savetxt("M_x.csv", M_x, delimiter=",")
    np.savetxt("K_x.csv", K_x, delimiter=",")

    print('Estimated Masses: ', m_est)
    print('Estimated Ks: ', k_est)

    fig1.savefig('figs/fig1.png')
    fig2.savefig('figs/fig2.png')
    fig3.savefig('figs/fig3.png')



if __name__ == '__main__':
    main()


