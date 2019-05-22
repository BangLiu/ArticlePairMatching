# coding:utf-8
from collections import OrderedDict
from scipy.linalg import norm


def sort_dict_by_key_str(d):
    """
    Sort dictionary by its key values.
    """
    return OrderedDict(
        sorted(d.items(), key=lambda t: t[0]))


def cosine_sim(a, b):
    if len(b) < len(a):
        a, b = b, a
    res = 0
    for key, a_value in a.items():
        res += a_value * b.get(key, 0)
    if res == 0:
        return 0
    try:
        res = res / (norm(list(a.values())) * norm(list(b.values())))
    except ZeroDivisionError:
        res = 0
    return res
