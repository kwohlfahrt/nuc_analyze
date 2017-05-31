from nuc_analyze.util import *

def test_flatten_dict():
    data = {0: {1: {2: 'foo', 3: 'foo2'}, 'bar': 'baz'}, 5: 'foo'}
    expected = {(0, 1, 2): 'foo', (0, 1, 3): 'foo2', (0, 'bar'): 'baz', (5,): 'foo'}

    assert expected == flatten_dict(data)


def test_unflatten_dict():
    data = {(0, 1, 2): 'foo', (0, 1, 3): 'foo2', (0, 'bar'): 'baz', (5,): 'foo'}
    expected = {0: {1: {2: 'foo', 3: 'foo2'}, 'bar': 'baz'}, 5: 'foo'}

    assert expected == unflatten_dict(data.items())
