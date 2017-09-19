import numpy as np
from h5py import File as HDFFile
from click.testing import CliRunner

nucs = [{'structures': {'0': {
    'coords': {
        '1': np.array([[[0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [1, 0, 0]]]),
        'X': np.array([[[0, 0, 0], [0, 0, 0]],
                       [[0, 2, 0], [0, 1, 0]]]),
    },
    'particles': {
        '1': {'positions': np.array([10, 200]),},
        'X': {'positions': np.array([0, 100]),},
    },
}}}, {'structures': {'0': {
    'coords': {
        '1': np.array([[[0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [1, 0, 0]]]),
        'X': np.array([[[0, 0, 0]],
                       [[0, 4, 0]]]),
    },
    'particles': {
        '1': {'positions': np.array([10, 200]),},
        'X': {'positions': np.array([100]),},
    },
}}}]

def test_rmsd():
    from nuc_analyze.rmsd import rmsd

    expected = [('1', [10, 200], [0., 0.5]), ('X', [0, 100], [1.0, 0.5])]
    np.testing.assert_equal(sorted(rmsd(nucs[0])), sorted(expected))

def test_rmsd_cli(tmpdir):
    from nuc_analyze.main import cli

    files = [tmpdir.join("test1.nuc"), tmpdir.join("test2.nuc")]

    for p, nuc in zip(files, nucs):
        with HDFFile(p, 'w') as f:
            for chromo, coords in sorted(nuc['structures']['0']['coords'].items()):
                f.create_dataset('structures/0/coords/{}'.format(chromo), data=coords)
            for chromo, particle in nuc['structures']['0']['particles'].items():
                f.create_dataset(
                    'structures/0/particles/{}/positions'.format(chromo),
                    data=particle['positions']
                )

    expected = ["1:10 0.0", "1:200 0.5", "X:100 2.0"]

    runner = CliRunner()
    result = runner.invoke(cli, ["rmsd", str(p)])
    assert result.exit_code == 0
    assert result.output == '\n'.join(expected) + '\n'
