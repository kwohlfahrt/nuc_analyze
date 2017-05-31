#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import operator as op
import click
from coherent_point_drift.least_squares import align
from coherent_point_drift.geometry import rigidXform
from functools import partial
from itertools import repeat

from .main import cli

def align_models(coords):
    ref_stack = np.concatenate([v[0] for v in coords.values()], axis=0)

    n_models = len(next(iter(coords.values())))
    ndim = next(iter(coords.values())).shape[-1]
    # Fitting to self returns NaN
    yield (np.eye(ndim), np.zeros(ndim), 1.)
    for i in range(1, n_models):
        model_stack = np.concatenate([v[i] for v in coords.values()], axis=0)
        yield align(ref_stack, model_stack)

def rmsd(nuc, structure="0", align=False):
    xforms = list(align_models(nuc['structures'][structure]['coords']))
    for chromo, coords in nuc['structures'][structure]['coords'].items():
        if align:
            coords = np.stack(
                [rigidXform(c, *xform) for c, xform in zip(coords, xforms)], axis=0
            )
        mean = np.mean(coords, axis=0, keepdims=True)
        error = np.linalg.norm((coords - mean), axis=-1)
        rmsds = np.sqrt(np.mean(error ** 2, axis=0))
        positions = nuc['structures'][structure]['particles'][chromo]['positions']
        yield chromo, positions, rmsds

@cli.command("rmsd")
@click.argument("nucs", type=Path, nargs=-1, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
@click.option("--position", multiple=True, type=(str, int),
              help="Which positions to look at (or all if none provided)")
@click.option("--align/--no-align", default=True, help="Align all models to the first")
def output_rmsd(nucs, structure, position, align):
    from functools import partial

    nucs_rmsds = []
    for nuc in nucs:
        with HDFFile(nuc, "r") as f:
            nuc_rmsds = {}
            for chromo, positions, rmsds in rmsd(f, structure, align):
                nuc_rmsds.update(zip(zip(repeat(chromo), positions), rmsds))
            nucs_rmsds.append(nuc_rmsds)
    conserved = set.intersection(*map(set, nucs_rmsds))
    if position:
        positions = filter(partial(op.contains, conserved), position)
    else:
        positions = sorted(conserved)
    for chromo, pos in positions:
        print("{}:{} {}".format(chromo, pos, max(rmsd[chromo, pos] for rmsd in nucs_rmsds)))
