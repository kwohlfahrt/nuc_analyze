#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import operator as op
import click
from coherent_point_drift.least_squares import align
from functools import partial
from itertools import repeat, chain
from scipy.cluster import hierarchy

from .main import cli

def distance(x, y):
    x = x.reshape(-1, 3)
    y = y.reshape(-1, 3)
    return np.sqrt(np.mean((y - x) ** 2))

@cli.command()
@click.argument("nuc", type=Path, required=True)
@click.option("--output", help="Where to save the plot")
@click.option("--figsize", type=(float, float), default=(8, 6),
              help="The size of the figure (in inches)")
@click.option("--structure", default="0", help="Which structure in the file to read")
def plot_clusters(nuc, output, figsize, structure):
    import matplotlib
    if output is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    with HDFFile(nuc, "r") as f:
        coords = np.concatenate(
            list(f['structures'][structure]['coords'].values()), axis=1
        )
        Z = hierarchy.linkage(
            coords.reshape(coords.shape[0], -1), method='single',
            metric=distance
        )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hierarchy.dendrogram(Z, ax=ax, link_color_func=lambda i: colors[0])
    ax.set_xlabel("model")
    ax.set_ylabel("RMSD")

    if output is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(output))
