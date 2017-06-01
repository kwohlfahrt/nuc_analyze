from collections import defaultdict

def flatten_dict(d):
    r = {}
    for key, value in d.items():
        try:
            r.update({(key,) + k: v for k, v in flatten_dict(value).items()})
        except AttributeError:
            r[(key,)] = value
    return r

def tree():
    def tree_():
        return defaultdict(tree_)
    return tree_()

def unflatten_dict(it):
    r = tree()

    for ks, v in it:
        d = r
        for k in ks[:-1]:
            d = d[k]
        d[ks[-1]] = v
    return r

def ceil_div(x, y):
    return x // y + (x % y != 0)
