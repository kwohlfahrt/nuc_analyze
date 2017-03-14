from pathlib import Path
from h5py import File as HDFFile
import numpy as np

def scale(nuc, structure="0"):
    coords = np.concatenate(list(nuc['structures'][structure]['coords'].values()), axis=1)
    return np.mean(np.std(coords, axis=1))

def main(args=None):
    from argparse import ArgumentParser
    from sys import argv

    parser = ArgumentParser(description="Analyze a .nuc file")
    parser.add_argument("nuc", type=Path, help="The file to analyze")
    parser.add_argument("--structure", type=str, default="0",
                        help="Which structure in the file to use")

    args = parser.parse_args(argv[1:] if args is None else args)

    with HDFFile(args.nuc, "r") as f:
        print("std: {}".format(scale(f, args.structure)))

if __name__ == "__main__":
    main()
