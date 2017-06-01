from pathlib import Path
from h5py import File as HDFFile
import numpy as np
from collections import Counter, defaultdict
from itertools import chain, product as cartesian
import click

from .main import cli
from .util import flatten_dict

def scale(nuc, structure="0"):
    coords = np.concatenate(list(nuc['structures'][structure]['coords'].values()), axis=1)
    return np.mean(np.std(coords, axis=1), axis=1)

def violations(nuc, structure="0", padding=0.0):
    coords = nuc['structures'][structure]['coords']
    restraints = nuc['structures'][structure]['restraints']
    violations = Counter()

    for (chr_a, chr_b), restraints in flatten_dict(restraints).items():
        if not len(restraints) > 0:
            continue
        a_coords = coords[chr_a][:][:, restraints['indices'][:, 0]]
        b_coords = coords[chr_b][:][:, restraints['indices'][:, 1]]
        dist = np.linalg.norm(a_coords - b_coords, axis=-1)
        viol = ((restraints['dists'][:, 1] * (1.0 + padding) < dist) |
                (restraints['dists'][:, 0] * (1.0 - padding) > dist))
        for model, model_violations in enumerate(viol):
            violations[model] += model_violations.sum()
    # Order of models doesn't matter, summarized anyway
    return list(violations.values())

@cli.command()
@click.argument("nucs", type=Path, nargs=-1, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
@click.option("--param", multiple=True, help="Which calculation parameters to print")
@click.option("--violation-padding", type=float, default=0.0,
              help="How much (relative) a restraint may be violated by")
def stats(nucs, structure, param, violation_padding):
    import csv
    from sys import stdout

    stat_names = {"scale": {}, "violations": {'padding': violation_padding}}
    stat_cols = list(chain.from_iterable(
        ("{}_mean".format(s), "{}_std".format(s)) for s in sorted(stat_names)
    ))

    writer = csv.DictWriter(stdout, ["filename"] + stat_cols + list(param))
    writer.writeheader()
    for nuc in nucs:
        with HDFFile(nuc, "r") as f:
            params = f['structures'][structure]['calculation'].attrs
            params = {k: params[k] for k in param}
            if "particle_sizes" in params:
                params["particle_sizes"] = params["particle_sizes"][-1]
            params["filename"] = str(nuc.name)

            for stat, kwargs in sorted(stat_names.items()):
                stat_values = globals()[stat](f, structure, **kwargs)
                params["{}_mean".format(stat)] = np.mean(stat_values)
                params["{}_std".format(stat)] = np.std(stat_values)
            writer.writerow(params)

@cli.command()
@click.argument("nucs", type=Path, nargs=-1, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
@click.option("--param", help="Which calculation parameter to plot against")
@click.option("--violation-padding", type=float, default=0.0,
              help="How much (relative) a restraint may be violated by")
def plot_stats(nucs, structure, param, violation_padding):
    import matplotlib.pyplot as plt

    stat_names = {"scale": {}, "violations": {'padding': violation_padding}}
    stats = defaultdict(lambda: np.empty((len(nucs), 3), dtype='float'))

    fig, axs = plt.subplots(len(stat_names), 1)

    for (i, nuc), (stat, kwargs) in cartesian(enumerate(nucs), stat_names.items()):
        with HDFFile(nuc, "r") as f:
            param_value = f['structures'][structure]['calculation'].attrs[param]
            if param_value == "particle_sizes":
                param_value["particle_sizes"] = param_value["particle_sizes"][-1]
            stat_values = globals()[stat](f, structure, **kwargs)
            stats[stat][i] = [param_value, np.mean(stat_values), np.std(stat_values)]

    for ax, (stat_name, data) in zip(axs, stats.items()):
        ax.set_ylabel(stat_name)
        ax.set_xlabel(param)
        data = np.sort(data, axis=0)
        ax.errorbar(data.T[0], data.T[1], yerr=data.T[2])
    fig.tight_layout()
    plt.show()
