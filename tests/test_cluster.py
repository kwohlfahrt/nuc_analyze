import numpy as np
import pytest
import h5py

from nuc_analyze.cluster import *

@pytest.fixture()
def runner():
    from click.testing import CliRunner
    return CliRunner()

@pytest.fixture()
def nucfile(tmpdir):
    filename = tmpdir.join("test.nuc")
    coords_dict = {
        'chr1': np.array([[
            [ 1., 0., 0.],
            [ 5., 0., 0.],
            [ 8., 0., 1.],
            [ 8., 1., 1.],
        ], [
            [ 1., 1., 0.],
            [ 5., 1., 0.],
            [ 8., 1., 1.],
            [ 8., 2., 1.],
        ]], dtype='double'),
        'chr2': np.array([[
            [-1., 0., 3.],
            [-5., 0., 3.],
            [-0., 0., 3.],
            [-0., 1., 3.],
        ], [
            [-1., 1., 3.],
            [-5., 1., 3.],
            [-0., 1., 3.],
            [-0., 2., 3.],
        ]], dtype='double'),
    }

    with h5py.File(filename, "w") as f:
        structure = f.create_group("structures/0")
        coords = structure.create_group("coords")
        for chromosome, data in coords_dict.items():
            coords.create_dataset(chromosome, data=data)
    return filename

def test_commandline(nucfile, tmpdir, runner):
    args = [str(nucfile), "--output", str(tmpdir.join("foo.pdf"))]
    result = runner.invoke(plot_clusters, args)
    assert result.exit_code == 0
