#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import operator as op
from functools import partial
import click
from coherent_point_drift.least_squares import align as least_squares

from .main import cli

@cli.command()
@click.argument("nuc", type=Path, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
def align(nuc, structure):
    with HDFFile(nuc, "r+") as f:
        coordss = f['structures'][structure]['coords']
        all_coords = np.concatenate(list(coordss.values()), axis=1)
        ref = np.median(all_coords, axis=0)
        xforms = list(map(partial(least_squares, ref), all_coords))
        for chr, coords in coordss.items():
            coords[:] = np.array(list(map(op.matmul, xforms, coords)), dtype=coords.dtype)
