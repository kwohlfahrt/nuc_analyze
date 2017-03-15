from pathlib import Path
from h5py import File as HDFFile
import numpy as np

def density(nuc, radius=1.0, structure="0"):
    params = nuc['structures'][structure]['calculation'].attrs
    radius = params['particle_sizes'][-1] ** params['scaling_exponent']

    coords = np.concatenate(list(nuc['structures'][structure]['coords'].values()), axis=1)
    neighbours = np.empty((coords.shape[0], coords.shape[1]), dtype='int')
    for i, structure in enumerate(coords):
        for j, coord in enumerate(structure):
            neighbours[i, j] = np.count_nonzero(np.linalg.norm(structure - coord, axis=-1) < radius)

    return np.mean(neighbours), np.std(neighbours)

def scale(nuc, structure="0"):
    coords = np.concatenate(list(nuc['structures'][structure]['coords'].values()), axis=1)
    return np.mean(np.std(coords, axis=1))

def main(args=None):
    from argparse import ArgumentParser
    from sys import argv, stdout
    import csv

    parser = ArgumentParser(description="Analyze a .nuc file")
    parser.add_argument("nucs", type=Path, nargs='+', help="The files to analyze")
    parser.add_argument("--structure", type=str, default="0",
                        help="Which structure in the file to use")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Number of bead-radii to consider for density")
    parser.add_argument("--params", type=str, nargs='*', default=[],
                        help="Which calculation parameters to print")

    args = parser.parse_args(argv[1:] if args is None else args)

    writer = csv.DictWriter(stdout, [
        "filename", "density_radius", "std", "density_mean", "density_std"
    ] + args.params)
    writer.writeheader()
    for nuc in args.nucs:
        with HDFFile(nuc, "r") as f:
            params = f['structures'][args.structure]['calculation'].attrs
            params = {k: params[k] for k in args.params}
            if "particle_sizes" in params:
                params["particle_sizes"] = params["particle_sizes"][-1]

            params["filename"] = str(nuc.name).split(".")[0]
            params["density_radius"] = args.radius

            params["std"] = scale(f, args.structure)
            params["density_mean"], params["density_std"] = density(f, args.radius, args.structure)
            writer.writerow(params)

if __name__ == "__main__":
    main()
