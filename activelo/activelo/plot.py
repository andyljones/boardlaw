import matplotlib.pyplot as plt
import numpy as np

def plot(soln):
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)

    ax = axes[0, 0]
    ax.errorbar(
        np.arange(soln.μd.shape[0]), 
        soln.μd[:, 0], yerr=soln.σd[0, :], marker='.', linestyle='')
    ax.set_xlim(0, len(soln.μ)-1)
    ax.set_title('μ v. first agent')

    ax = axes[0, 1]
    ax.imshow(np.where(soln.σd > 0, soln.σd, np.nan))
    ax.set_title('σd')

    ax = axes[1, 0]
    ax.plot(soln.trace.l)
    ax.set_xlim(0, len(soln.trace.l)-1)
    ax.set_title('loss')
    ax.set_yscale('log')

    ax = axes[1, 1]
    ax.plot(soln.trace.relnorm)
    ax.set_xlim(0, len(soln.trace.relnorm)-1)
    ax.set_yscale('log')
    ax.set_title('norms')

    return fig