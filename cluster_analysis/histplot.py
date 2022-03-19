import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize': (4.8, 4)})

from matplotlib.colors import LogNorm, Normalize

import os

from collections import defaultdict


def plot_hist(path, 
              hist_file, 
              main_str, 
              aux_str, 
              fig_name, 
              fig_height, 
              fig_width, 
              colorbar=True,
              text=True,
              n_max=-1, 
              v_min=0., 
              v_max=1.):

    """
    Function that plots the histogram file

    Args:

    path (string): path of the histogram text file
    hist_file (string): name of the histogram text file
    main_str (string): string of the main ion
    aux_str (string): string of the aux ion
    fig_name (string): name of the plotted figure file
    fig_height (float): height of the figure
    fig_width (float): width of the figure
    colorbar (Boolean): whether to plot colorbar
    n_max (int): maximum number of main / aux ions shown on the plot
    v_min (float): lower limit of the colorbar
    v_max (float): upper limit of the colorbar
    """

    os.chdir(path)

    n_main_max = 0
    n_aux_max = 0

    total_num = 0

    hist_dict = defaultdict(lambda: 0)

    if n_max < 0:
        n_thresh = 10000
    else:
        n_thresh = n_max

    with open(hist_file) as f:

        for line in f:

            n_main, n_aux = line.strip().split(',')

            n_main = int(n_main)
            n_aux = int(n_aux)

            if (n_main <= n_thresh) and (n_aux <= n_thresh):

                n_main_max = max(n_main, n_main_max)
                n_aux_max = max(n_aux, n_aux_max)
    
                # update corresponding entry in hist_dict
    
                hist_dict[(n_main, n_aux)] += 1
                total_num += 1

    if n_max >= 0:
        n = n_max
    else:
        n = max(n_main_max, n_aux_max) + 1

    ## create histogram frequencies

    hist_freq = np.zeros((n+1, n+1))

    for key in hist_dict.keys():

        n_main, n_aux = key
        print(key)
        hist_freq[n_main, n_aux] = hist_dict[key] / total_num

    # plot 2D grid data
    fig = plt.figure(figsize=(fig_height, fig_width))
    ax = fig.add_subplot(111)

    ax.set_xticks(range(0, n+1, 1))
    ax.set_yticks(range(0, n+1, 1))
    
    ax.set_xlim((-0.5, n+0.5))
    ax.set_ylim((-0.5, n+0.5))
    
    ax.set_xlabel("Number of "+main_str)
    ax.set_ylabel("Number of "+aux_str)
    
    img = ax.imshow(hist_freq,
                    # norm=LogNorm(vmin=0., vmax=1.),
                    cmap=plt.get_cmap('plasma'),
                    vmin=v_min, vmax=v_max)

    if colorbar:
        fig.colorbar(img, cmap=plt.get_cmap('plasma'))
        
    # Loop over data dimensions and create text annotations.
    if text:
        for i in range(n+1):
            for j in range(n+1):
                text = ax.text(j, i, "{:.2f}".format(hist_freq[i, j]),
                                ha="center", va="center", color="w")
    
    plt.tight_layout()

    if isinstance(fig_name, str):
        if len(fig_name) > 0:
            fig.savefig(fig_name+'.svg', dpi=150)

    plt.show()

