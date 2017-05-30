from pathlib import Path
from h5py import File as HDFFile
import numpy as np
from collections import Counter
from itertools import chain
import click

from .main import cli

def flatten_dict(d):
  r = {}
  for key, value in d.items():
    try:
      r.update({(key,) + k: v for k, v in flatten_dict(value).items()})
    except AttributeError:
      r[(key,)] = value
  return r

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

    stat_names = [("scale", {}), ("violations", {'padding': violation_padding})]
    stat_cols = list(chain.from_iterable(
        ("{}_mean".format(s), "{}_std".format(s)) for s, _ in stat_names
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

            for stat, kwargs in stat_names:
                stat_values = globals()[stat](f, structure, **kwargs)
                params["{}_mean".format(stat)] = np.mean(stat_values)
                params["{}_std".format(stat)] = np.std(stat_values)
            writer.writerow(params)
