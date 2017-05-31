import numpy as np
from h5py import File as HDFFile
from click.testing import CliRunner

# Structure with two chromosomes x two models x two particles
# Add 'foo' as a test attr
Restraint = np.dtype([('indices', 'int', 2), ('dists', 'float', 2)])
nuc = {'structures': {'0': {
    'coords': {
        '1': np.array([[[0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [1, 1, 1]]]),
        'X': np.array([[[0, 0, 0], [0, 0, 0]],
                       [[2, 2, 2], [1, 1, 1]]]),
    },
    'restraints': {'1': {
        '1': np.array([((0, 0), (0.5, 1.0)),
                       ((0, 1), (0.0, 0.1))], dtype=Restraint),
        'X': np.array([((0, 0), (0.0, 1.0)),
                       ((1, 0), (0.0, 0.1))], dtype=Restraint),
    }},
    'calculation': {'attrs': {'foo': 1}}
}}}

def test_scale():
    from nuc_analyze.stats import scale
    from math import sqrt

    expected = np.array([0.0, sqrt(2) / 2])
    np.testing.assert_equal(scale(nuc), expected)

def test_violations():
    from nuc_analyze.stats import violations


    assert set(violations(nuc)) == {1, 4}
    violations = violations(nuc)

def test_csv(tmpdir):
    from nuc_analyze.main import cli
    from nuc_analyze.stats import flatten_dict
    from math import sqrt
    p = tmpdir.join("test.nuc")

    with HDFFile(p, 'w') as f:
        for chromo, coords in nuc['structures']['0']['coords'].items():
            f.create_dataset('structures/0/coords/{}'.format(chromo), data=coords)
        flat_restraints = flatten_dict(nuc['structures']['0']['restraints'])
        for (chr_a, chr_b), restraints in flat_restraints.items():
            f.create_dataset(
                'structures/0/restraints/{}/{}'.format(chr_a, chr_b), data=restraints
            )
        calculation = f.create_group('structures/0/calculation')
        for attr, v in nuc['structures']['0']['calculation']['attrs'].items():
            calculation.attrs[attr] = v

    scales = [0.0, sqrt(2) / 2]
    violations = [1, 4]
    expected = [
        'test.nuc', np.mean(scales), np.std(scales), np.mean(violations),
        np.std(violations), 1
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ["stats", str(p), "--param", "foo"])
    assert result.exit_code == 0
    out = iter(result.output.splitlines())
    assert next(out) == 'filename,scale_mean,scale_std,violations_mean,violations_std,foo'
    assert next(out) == ','.join(map(str, expected))
