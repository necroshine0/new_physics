import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(X_real, C_real, X_fake, C_fake):
    CX_real = np.concatenate((C_real, X_real), axis=1)
    CX_fake = np.concatenate((C_fake, X_fake), axis=1)
    x_names = [r'$m_{jj}$ (TeV)', r'$m_{j1}$ (TeV)', r'$m_{j2}-m_{j1}$ (TeV)', r'$\tau_{21}^{j1}$', r'$\tau_{21}^{j2}$']

    plt.figure(figsize=(12, 15))
    for i in range(CX_real.shape[1]):
        j = (i + 1) % 5
        plt.subplot(3, 2, i + 1)
        bins = np.linspace(CX_real[:, j].min(), CX_real[:, j].max(), 101)
        plt.hist(CX_fake[:, j], bins=bins, density=True, label='Generated samples',
                 histtype='bar', color='cornflowerblue', alpha=0.6)
        plt.hist(CX_real[:, j], bins=bins, density=True, label='Real samples',
                 histtype='step', color='C3', linewidth=1.5)
        plt.xlabel(x_names[j], size=14)
        plt.ylabel('Events (a.u.)', size=14)
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.legend()
    plt.tight_layout()
    plt.show()
