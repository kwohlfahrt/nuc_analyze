#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import operator as op
import click
from coherent_point_drift.least_squares import align
from coherent_point_drift.geometry import rigidXform
from functools import partial
from itertools import repeat, chain
from scipy.cluster import hierarchy

from .main import cli

def aligned_distance(x, y):
    x = x.reshape(-1, 3)
    y = y.reshape(-1, 3)

    y = rigidXform(y, *align(x, y))
    return np.sqrt(np.mean((y - x) ** 2))

@cli.command()
@click.argument("nuc", type=Path, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
def plot_clusters(nuc, structure):
    import matplotlib.pyplot as plt

    with HDFFile(nuc, "r") as f:
        coords = np.concatenate(
            list(f['structures'][structure]['coords'].values()), axis=1
        )
        Z = hierarchy.linkage(
            coords.reshape(coords.shape[0], -1), method='single',
            metric=aligned_distance
        )
    fig, ax = plt.subplots(1, 1)
    hierarchy.dendrogram(Z, ax=ax)
    ax.set_xlabel("model")
    plt.show()
