from pathlib import Path
from h5py import File as HDFFile
import numpy as np
from collections import Counter
from itertools import chain

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

def violations(nuc, structure="0"):
    coords = nuc['structures'][structure]['coords']
    restraints = nuc['structures'][structure]['restraints']
    violations = Counter()

    for (chr_a, chr_b), restraints in flatten_dict(restraints).items():
        if not len(restraints) > 0:
            continue
        a_coords = coords[chr_a][:][:, restraints['indices'][:, 0]]
        b_coords = coords[chr_b][:][:, restraints['indices'][:, 1]]
        dist = np.linalg.norm(a_coords - b_coords, axis=-1)
        viol = ((restraints['dists'][:, 1] < dist) |
                (restraints['dists'][:, 0] > dist))
        for model, model_violations in enumerate(viol):
            violations[model] += model_violations.sum()
    # Order of models doesn't matter, summarized anyway
    return list(violations.values())

def main(args=None):
    from argparse import ArgumentParser
    from sys import argv, stdout
    import csv

    parser = ArgumentParser(description="Analyze a .nuc file")
    parser.add_argument("nucs", type=Path, nargs='+', help="The files to analyze")
    parser.add_argument("--structure", type=str, default="0",
                        help="Which structure in the file to use")
    parser.add_argument("--params", type=str, nargs='*', default=[],
                        help="Which calculation parameters to print")

    args = parser.parse_args(argv[1:] if args is None else args)

    stat_names = ["scale", "violations"]
    stat_cols = list(chain.from_iterable(
        ("{}_mean".format(s), "{}_std".format(s)) for s in stat_names
    ))

    writer = csv.DictWriter(stdout, ["filename"] + stat_cols + args.params)
    writer.writeheader()
    for nuc in args.nucs:
        with HDFFile(nuc, "r") as f:
            params = f['structures'][args.structure]['calculation'].attrs
            params = {k: params[k] for k in args.params}
            if "particle_sizes" in params:
                params["particle_sizes"] = params["particle_sizes"][-1]
            params["filename"] = str(nuc.name)

            for stat in stat_names:
                stat_values = globals()[stat](f, args.structure)
                params["{}_mean".format(stat)] = np.mean(stat_values)
                params["{}_std".format(stat)] = np.std(stat_values)
            writer.writerow(params)

if __name__ == "__main__":
    main()
