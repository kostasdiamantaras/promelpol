import numpy as np

from functools import reduce, partial
from multiprocessing import Pool


def pmap(f, col):
    pool = Pool(5)
    result = pool.map(f, col)
    pool.close()
    return result


def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


arrayl = compose(np.array, list)


def mapt(f, col):
    return compose(tuple, partial(map, f))(col)

def mapnp(f, col):
    return compose(arrayl, partial(map, f))(col)


def countif(f, col):
    return compose(len, arrayl, partial(filter, f))(col)


def append(acc, col):
    return np.append(acc, col, axis=0)


def enqueue(acc, l):
    acc.insert(len(acc), l)
    return acc


def add(x, y):
    return x + y


def identity(x):
	return x


def flatten(x):
    return reduce(append, x)


def add_key(dic, key, val):
    dic[key] = val
    return dic
