#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import click

from .main import cli

def rmsd(nuc, structure="0"):
    from itertools import repeat

    for chromo, coords in nuc['structures'][structure]['coords'].items():
        mean = np.mean(coords, axis=0, keepdims=True)
        error = np.linalg.norm((coords - mean), axis=-1)
        rmsds = np.sqrt(np.mean(error ** 2, axis=0))
        positions = nuc['structures'][structure]['particles'][chromo]['positions']
        positions = zip(repeat(chromo), positions)
        yield from zip(positions, rmsds)

@cli.command("rmsd")
@click.argument("nucs", type=Path, nargs=-1, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
def output_rmsd(nucs, structure):
    from itertools import starmap

    rmsds = []
    for nuc in nucs:
        with HDFFile(nuc, "r") as f:
            rmsds.append(dict(rmsd(f, structure)))
    conserved = set.intersection(*map(set, rmsds))
    for chromo, pos in sorted(conserved):
        print("{}:{} {}".format(chromo, pos, max(rmsd[chromo, pos] for rmsd in rmsds)))
