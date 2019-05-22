# -*- coding: utf-8 -*-
import os
import sys
import pickle


def replace_sep(fin, fout, sep_ini, sep_fin):
    """
    Replace delimiter in a file.
    """
    fin = open(fin, "r")
    fout = open(fout, "w")
    for line in fin:
        fout.write(line.replace(sep_ini, sep_fin))
    fin.close()
    fout.close()


def remove_quotes(fin, fout):
    """
    Remove quotes in lines.
    If a line has odd number quotes, remove all quotes in this line.
    """
    fin = open(fin)
    fout = open(fout, "w")
    for line in fin:
        fout.write(line.replace("\"", ""))
    fin.close()
    fout.close()


def pickle_dump_large_file(obj, filepath):
    """
    This is a defensive way to write pickle.write,
    allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load_large_file(filepath):
    """
    This is a defensive way to write pickle.load,
    allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
